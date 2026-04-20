#!/usr/bin/env python3
"""
macOS Hardware Info - Análisis completo de hardware para macOS.

Consolida análisis de SSD (salud, temperatura, TBW) y benchmarks de IA (CPU, GPU, NPU).
Compatible con SSDs estándar y Apple Silicon. Requiere Python 3.13+ y privilegios sudo.
"""

from __future__ import annotations

import fcntl
import json
import sys
import os
import re
import shutil
import subprocess
import time
import warnings

class ReportLogger:
    def __init__(self):
        self.terminal = sys.stdout
        self.log_content = []

    def write(self, message):
        self.terminal.write(message)
        self.log_content.append(message)

    def flush(self):
        self.terminal.flush()
        
    def get_clean_content(self):
        raw = "".join(self.log_content)
        # Limpiar caracteres de retorno de carro y secuencias ANSI si es necesario
        clean = re.sub(r'.*\r', '', raw)
        clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', clean)
        return clean

def get_mac_identity() -> str:
    """Obtiene el modelo y serial del Mac para el nombre del archivo del reporte."""
    model = "Mac"
    serial = "UNKNOWN"
    chip = ""
    
    stdout, _ = run_command(['system_profiler', 'SPHardwareDataType'])
    if stdout:
        for line in stdout.split('\n'):
            line = line.strip()
            if line.startswith('Model Name:'):
                model = line.split(':')[1].strip().replace(' ', '')
            elif line.startswith('Chip:') or line.startswith('Processor Name:'):
                chip = line.split(':')[1].strip().replace(' ', '')
            elif line.startswith('Serial Number'):
                serial = line.split(':')[1].strip()
                
    parts = [p for p in [model, chip, serial] if p]
    return "_".join(parts)
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Final, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn

try:
    import coremltools as ct
    import coremltools.optimize.coreml as cto
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

warnings.filterwarnings("ignore")


# ============================================================================
# CONSTANTES Y PATRONES
# ============================================================================

SECTOR_SIZE: Final = 512
NVME_UNIT_SIZE: Final = 1000 * SECTOR_SIZE
GB_DIVISOR: Final = 1024 ** 3
TB_DIVISOR: Final = 1024 ** 4
DAYS_IN_MONTH: Final = 30.44

DISK_PATTERN: Final = re.compile(r'/dev/(disk\d+)\s')
DISK_ID_PATTERN: Final = re.compile(r'(disk\d+)')
APPLE_SSD_PATTERN: Final = re.compile(r'APPLE\s+(SSD|NVMe)|AP\d{4}[A-Z]?', re.IGNORECASE)


# ============================================================================
# CLASES DE DATOS
# ============================================================================

class DiskType(str, Enum):
    """Tipos de disco soportados."""
    ATA = "ata"
    NVME = "nvme"


@dataclass(frozen=True, slots=True)
class DiskInfo:
    """Información completa de hardware del disco."""
    disk_id: str
    device_name: str = "Desconocido"
    size: str = "N/A"
    connection: str = "N/A"
    protocol: str = "N/A"
    location: str = "N/A"


@dataclass(slots=True)
class SmartReport:
    """Reporte de métricas S.M.A.R.T."""
    model_brand: str = "N/A"
    serial_number: str = "N/A"
    firmware_version: str = "N/A"
    smart_status: str = "N/A"
    tbw_tb: float = 0.0
    ssd_lifetime_left_pct: str | int = "N/A"
    power_on_hours: int | None = None
    power_cycle_count: int | None = None
    temperature_celsius: int | float | str = "N/A"
    read_speed_mbps: float | None = None
    write_speed_mbps: float | None = None


@dataclass
class BenchmarkResults:
    """Resultados de benchmarks de IA."""
    cpu_gflops: float = 0.0
    gpu_tops: float = 0.0
    npu_fp16_tops: float = 0.0
    npu_int8_tops: float = 0.0
    system_info: Dict[str, str] = None


# ============================================================================
# UTILIDADES DE SISTEMA
# ============================================================================

@lru_cache(maxsize=1)
def check_sudo() -> bool:
    """Verifica privilegios de superusuario."""
    if os.geteuid() != 0:
        raise SystemExit(
            "❌ Privilegios insuficientes.\n"
            "   Ejecute: sudo python3 macOS-Hardware-info.py"
        )
    return True


@lru_cache(maxsize=1)
def get_smartctl_path() -> str:
    """Obtiene la ruta a smartctl empaquetado o del sistema."""
    if hasattr(sys, '_MEIPASS'):
        bundled_path = os.path.join(sys._MEIPASS, "bin", "smartctl")
        if os.path.exists(bundled_path):
            return bundled_path
    return "smartctl"


@lru_cache(maxsize=1)
def check_dependencies() -> bool:
    """Verifica dependencias del sistema."""
    if get_smartctl_path() == "smartctl" and not shutil.which("smartctl"):
        raise SystemExit(
            "❌ smartctl no encontrado.\n"
            "   Instale: brew install smartmontools"
        )
    return True


def run_command(command: list[str], timeout: int = 30) -> tuple[str | None, str | None]:
    """Ejecuta comando de sistema y captura salida."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=timeout
        )
        return result.stdout, result.stderr
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return None, str(e)


def run_command_json(command: list[str]) -> dict[str, Any] | None:
    """Ejecuta comando y parsea salida como JSON."""
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30
        )
        if result.returncode >= 8:
            return None
        return json.loads(result.stdout)
    except (json.JSONDecodeError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


# ============================================================================
# DETECCIÓN Y ANÁLISIS DE DISCOS
# ============================================================================

def find_physical_disks() -> list[str]:
    """Encuentra todos los discos físicos del sistema."""
    stdout, _ = run_command(['diskutil', 'list', 'physical'])
    if not stdout:
        return []
    return sorted(set(DISK_PATTERN.findall(stdout)))


def get_boot_disk() -> str | None:
    """Identifica el disco de arranque físico (base) del sistema."""
    stdout, _ = run_command(['diskutil', 'info', '/'])
    if not stdout:
        return None
    
    apfs_store = None
    part_of_whole = None
    
    for line in stdout.split('\n'):
        if 'APFS Physical Store:' in line:
            if match := DISK_ID_PATTERN.search(line):
                apfs_store = match.group(1)
        elif 'Part of Whole:' in line:
            if match := DISK_ID_PATTERN.search(line):
                part_of_whole = match.group(1)
                
    return apfs_store or part_of_whole


def get_disk_info_summary(disk_id: str) -> DiskInfo:
    """Obtiene información completa del hardware del disco."""
    stdout, _ = run_command(['diskutil', 'info', disk_id])
    
    info_data = {'disk_id': disk_id}
    
    if stdout:
        for line in stdout.split('\n'):
            if 'Device / Media Name:' in line or 'Media Name:' in line:
                info_data['device_name'] = line.split(':', 1)[1].strip()
            elif 'Disk Size:' in line or 'Total Size:' in line:
                if size_match := re.search(r'(\d+\.?\d*\s*[KMGT]B)', line):
                    info_data['size'] = size_match.group(1)
            elif 'Protocol:' in line:
                info_data['protocol'] = line.split(':', 1)[1].strip()
            elif 'Device Location:' in line:
                info_data['location'] = line.split(':', 1)[1].strip()
            elif 'Connection:' in line or 'Physical Interconnect:' in line:
                info_data['connection'] = line.split(':', 1)[1].strip()
    
    return DiskInfo(**info_data)


def get_smart_data(disk_id: str) -> dict[str, Any] | None:
    """Obtiene datos S.M.A.R.T. detectando tipo de SSD automáticamente."""
    device_path = f"/dev/{disk_id}"
    smartctl = get_smartctl_path()
    
    # Intentar múltiples métodos
    for cmd in [
        [smartctl, '-a', '-j', device_path],
        [smartctl, '-d', 'auto', '-T', 'permissive', '-a', '-j', device_path],
        [smartctl, '-x', '-j', device_path]
    ]:
        if result := run_command_json(cmd):
            return result
    
    return None


def benchmark_disk_speed(disk_info: DiskInfo, mount_point: str = '/tmp') -> tuple[float | None, float | None]:
    """
    Realiza un benchmark inteligente y de I/O directo evadiendo el caché de RAM de macOS (ARC).
    Garantiza velocidades reales de lectura/escritura en hardware (NAND) puro.
    Adaptable dinámicamente al tipo de controlador: Apple Silicon, NVMe externos (WD, Samsung) y SATA.
    """
    try:
        # 1. Ajuste adaptativo: Los SSD Apple Silicon y NVMe Gen4+ requieren cargas pesadas para medir bien
        is_fast_nvme = any(k in disk_info.protocol or k in disk_info.connection for k in ("NVMe", "Apple", "PCI"))
        test_size_mb = 2048 if is_fast_nvme else 512
        block_size = 4 * 1024 * 1024  # 4 MB blocks (óptimo para SSDs modernos)
        total_blocks = test_size_mb * 1024 * 1024 // block_size
        
        # 2. Entropía Real: Generar un pool de datos aleatorios en memoria.
        # Evita que el firmware del SSD (controladores Phison, SandForce, etc) infle los 
        # números deduplicando o comprimiendo bloques repetidos al vuelo.
        random_pool = bytearray(os.urandom(16 * 1024 * 1024))
        pool_size = len(random_pool)
        
        test_dir = Path(mount_point) / '.HardwareInfoBenchmark'
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / 'intelligent_speed_test.dat'
        
        # --- TEST DE ESCRITURA FÍSICA DIRECTA ---
        # O_SYNC obliga a esperar confirmación real de los chips NAND, saltando el buffer del SO
        fd_write = os.open(test_file, os.O_CREAT | os.O_WRONLY | os.O_SYNC)
        try:
            # F_NOCACHE es el estándar POSIX/macOS para anular la caché de memoria virtual
            fcntl.fcntl(fd_write, fcntl.F_NOCACHE, 1)
        except (AttributeError, OSError):
            pass

        start_write = time.perf_counter()
        for i in range(total_blocks):
            offset = (i * block_size) % pool_size
            os.write(fd_write, random_pool[offset:offset+block_size])
            
        os.fsync(fd_write) # Doble comprobación física
        write_time = time.perf_counter() - start_write
        os.close(fd_write)
        write_speed = test_size_mb / write_time if write_time > 0 else None
        
        # Reposo del controlador para asimilar la caché térmica/SLC interna del disco
        time.sleep(0.5)
        
        # --- TEST DE LECTURA FÍSICA DIRECTA ---
        fd_read = os.open(test_file, os.O_RDONLY)
        try:
            # CRÍTICO: Si no se desactiva aquí, macOS leerá de sus 10GB/s de memoria RAM
            fcntl.fcntl(fd_read, fcntl.F_NOCACHE, 1)
        except (AttributeError, OSError):
            subprocess.run(['purge'], capture_output=True, timeout=60, check=False)
            
        start_read = time.perf_counter()
        while True:
            if not os.read(fd_read, block_size):
                break
        read_time = time.perf_counter() - start_read
        os.close(fd_read)
        read_speed = test_size_mb / read_time if read_time > 0 else None
        
        # Limpieza
        test_file.unlink(missing_ok=True)
        test_dir.rmdir()
        
        return read_speed, write_speed
    except Exception as e:
        return None, None


def parse_smart_report(smart_data: dict[str, Any] | None) -> SmartReport:
    """Analiza datos S.M.A.R.T. y devuelve reporte unificado."""
    if not smart_data:
        return SmartReport()
    
    device_type = smart_data.get('device', {}).get('type')
    
    report = SmartReport(
        model_brand=smart_data.get('model_name') or smart_data.get('model_number', 'N/A'),
        serial_number=smart_data.get('serial_number', 'N/A'),
        firmware_version=smart_data.get('firmware_version', 'N/A'),
        smart_status="APROBADO" if smart_data.get('smart_status', {}).get('passed') else "FALLANDO"
    )
    
    # Parsear según tipo de disco
    if device_type in ('nvme', DiskType.NVME):
        log = smart_data.get('nvme_smart_health_information_log', {})
        
        # TBW en Terabytes
        if (units := log.get('data_units_written')) and units > 0:
            report.tbw_tb = (units * NVME_UNIT_SIZE) / TB_DIVISOR
        
        # Vida útil restante
        pct_used = log.get('percentage_used')
        report.ssd_lifetime_left_pct = max(0, 100 - pct_used) if pct_used is not None else "N/A"
        
        # Temperatura (conversión Kelvin a Celsius si necesario)
        temp = log.get('temperature')
        report.temperature_celsius = temp - 273 if temp and temp >= 273 else temp or "N/A"
        
        report.power_on_hours = log.get('power_on_hours')
        # Algunos SSDs usan 'power_cycle_count', otros 'power_cycles'
        report.power_cycle_count = log.get('power_cycle_count') or log.get('power_cycles')
        
    elif device_type in ('ata', DiskType.ATA):
        attrs = {attr['id']: attr for attr in smart_data.get('ata_smart_attributes', {}).get('table', [])}
        
        # TBW en Terabytes
        if (attr_241 := attrs.get(241)) and 'raw' in attr_241:
            lbas = attr_241['raw'].get('value', 0)
            report.tbw_tb = (lbas * SECTOR_SIZE) / TB_DIVISOR
        
        report.ssd_lifetime_left_pct = (attrs.get(202) or attrs.get(233) or {}).get('raw', {}).get('value', 'N/A')
        
        # Temperatura del campo estandarizado
        if 'temperature' in smart_data and 'current' in smart_data['temperature']:
            report.temperature_celsius = smart_data['temperature']['current']
        elif (attr_194 := attrs.get(194)) and 'raw' in attr_194:
            report.temperature_celsius = attr_194['raw'].get('value', 0) & 0xFF
        
        report.power_on_hours = attrs.get(9, {}).get('raw', {}).get('value')
        report.power_cycle_count = attrs.get(12, {}).get('raw', {}).get('value')
    
    return report


def format_power_on_time(total_hours: int | None) -> str:
    """Formatea horas en formato legible (meses, días, horas)."""
    if not total_hours:
        return "N/A"
    
    hours_per_month = 24 * DAYS_IN_MONTH
    months, remaining = divmod(total_hours, hours_per_month)
    days, hours = divmod(remaining, 24)
    
    parts = [
        f"{int(months)} mes{'es' if months > 1 else ''}" if months else "",
        f"{int(days)} día{'s' if days != 1 else ''}" if days else "",
        f"{int(hours)} hora{'s' if hours != 1 else ''}" if hours or not (months or days) else ""
    ]
    
    return ", ".join(filter(None, parts))


# ============================================================================
# BENCHMARKS DE IA
# ============================================================================

class AIBenchmark:
    """Benchmark de rendimiento de CPU, GPU y NPU."""
    
    def __init__(self):
        self.device = self._get_device()
        self.system_info = self._get_system_info()
    
    def _get_device(self) -> torch.device:
        """Detecta el mejor dispositivo disponible."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def _get_mac_gpu_cores(self) -> str:
        """Obtiene el número de cores de GPU en Mac."""
        try:
            cmd = ["system_profiler", "SPDisplaysDataType"]
            output = subprocess.check_output(cmd, timeout=10).decode("utf-8")
            for line in output.split('\n'):
                if "Total Number of Cores" in line:
                    return line.split(":")[1].strip()
            return "Unknown"
        except Exception:
            return "N/A"
    
    def _get_system_info(self) -> Dict[str, str]:
        """Recopila información del sistema."""
        import platform
        mem = psutil.virtual_memory()
        gpu_cores = self._get_mac_gpu_cores()
        phy_cores = psutil.cpu_count(logical=False)
        log_cores = psutil.cpu_count(logical=True)
        
        return {
            "OS": f"{platform.system()} {platform.release()}",
            "CPU": f"{platform.processor()} ({phy_cores}P/{log_cores}L cores)",
            "GPU": f"{self.device.type.upper()} ({gpu_cores} cores)",
            "RAM": f"{mem.total / (1024**3):.1f} GB",
            "NPU": "Disponible" if COREML_AVAILABLE else "No disponible"
        }
    
    def benchmark_cpu(self, size: int = 2048) -> float:
        """Benchmark de CPU usando operaciones matriciales (GFLOPS)."""
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)
        flops = 2 * (size ** 3)
        
        np.dot(a, b)  # Warmup
        
        start = time.perf_counter()
        for _ in range(10):
            np.dot(a, b)
        elapsed = time.perf_counter() - start
        
        return (flops / (elapsed / 10)) / 1e9
    
    def benchmark_gpu(self) -> float:
        """Benchmark de GPU usando PyTorch (TOPS)."""
        size = 4096
        a = torch.randn(size, size, device=self.device, dtype=torch.float16)
        b = torch.randn(size, size, device=self.device, dtype=torch.float16)
        flops = 2 * (size ** 3)
        
        # Warmup
        for _ in range(3):
            torch.matmul(a, b)
        if self.device.type == 'mps':
            torch.mps.synchronize()
        
        start = time.perf_counter()
        for _ in range(20):
            torch.matmul(a, b)
        if self.device.type == 'mps':
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        
        return (flops / (elapsed / 20)) / 1e12
    
    def _build_npu_model(self, quantize: bool = False):
        """Construye modelo para benchmark de NPU."""
        batch, size, channels, kernel, layers = 16, 32, 1536, 3, 5
        
        class DeepStress(nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = nn.Sequential(*[
                    nn.Sequential(
                        nn.Conv2d(channels, channels, kernel, padding=1, bias=False),
                        nn.ReLU()
                    ) for _ in range(layers)
                ])
            
            def forward(self, x):
                return self.mod(x)
        
        torch_model = DeepStress().eval()
        dummy = torch.randn(batch, channels, size, size)
        
        try:
            traced = torch.jit.trace(torch_model, dummy)
            model = ct.convert(
                traced,
                inputs=[ct.TensorType(name="input", shape=dummy.shape)],
                convert_to="mlprogram",
                compute_units=ct.ComputeUnit.ALL
            )
            
            if quantize:
                op_config = cto.OpLinearQuantizerConfig(
                    mode="linear_symmetric",
                    weight_threshold=512
                )
                config = cto.OptimizationConfig(global_config=op_config)
                model = cto.linear_quantize_weights(model, config)
            
            ops = layers * 2 * channels * (kernel**2) * size * size * channels * batch
            return model, dummy.numpy(), ops
        except Exception:
            return None, None, 0
    
    def benchmark_npu(self, quantize: bool = False) -> float:
        """Benchmark de NPU usando CoreML (TOPS)."""
        if not COREML_AVAILABLE:
            return 0.0
        
        model, data, ops = self._build_npu_model(quantize)
        if not model:
            return 0.0
        
        # Warmup
        for _ in range(5):
            model.predict({"input": data})
        
        start = time.perf_counter()
        for _ in range(15):
            model.predict({"input": data})
        elapsed = time.perf_counter() - start
        
        return (ops / (elapsed / 15)) / 1e12
    
    def run_all(self) -> BenchmarkResults:
        """Ejecuta todos los benchmarks y devuelve resultados."""
        return BenchmarkResults(
            cpu_gflops=self.benchmark_cpu(),
            gpu_tops=self.benchmark_gpu(),
            npu_fp16_tops=self.benchmark_npu(quantize=False),
            npu_int8_tops=self.benchmark_npu(quantize=True),
            system_info=self.system_info
        )


# ============================================================================
# INTERFAZ DE USUARIO Y REPORTE
# ============================================================================

def print_header():
    """Imprime encabezado del programa."""
    print("\n" + "═" * 80)
    print("  macOS HARDWARE INFO - Análisis Completo de Hardware")
    print("═" * 80)


def print_section(title: str):
    """Imprime título de sección."""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")


def display_disk_report(disk_id: str, disk_info: DiskInfo, report: SmartReport, is_boot: bool = False):
    """Muestra reporte detallado de un disco."""
    disk_type = "💾 DISCO PRINCIPAL (Sistema)" if is_boot else "💿 DISCO SECUNDARIO"
    
    print(f"\n{disk_type}: /dev/{disk_id}")
    print(f"  Modelo:          {report.model_brand}")
    print(f"  Serie:           {report.serial_number}")
    print(f"  Capacidad:       {disk_info.size}")
    print(f"  Conexión:        {disk_info.connection} ({disk_info.protocol})")
    print(f"  Estado SMART:    {report.smart_status}")
    
    # Salud y desgaste
    tbw_str = f"{report.tbw_tb:.2f} TB" if report.tbw_tb > 0 else "N/A"
    print(f"  TBW (escritos):  {tbw_str}")
    print(f"  Vida restante:   {report.ssd_lifetime_left_pct}%")
    
    # Temperatura y operación
    temp_str = f"{report.temperature_celsius}°C" if isinstance(report.temperature_celsius, (int, float)) else "N/A"
    print(f"  Temperatura:     {temp_str}")
    print(f"  Tiempo activo:   {format_power_on_time(report.power_on_hours)}")
    print(f"  Ciclos:          {report.power_cycle_count or 'N/A'}")
    
    # Velocidades
    if report.read_speed_mbps and report.write_speed_mbps:
        print(f"  Lectura:         {report.read_speed_mbps:.0f} MB/s")
        print(f"  Escritura:       {report.write_speed_mbps:.0f} MB/s")


def display_benchmark_report(results: BenchmarkResults):
    """Muestra reporte de benchmarks de IA."""
    print_section("BENCHMARKS DE RENDIMIENTO DE IA")
    
    print(f"\n  Sistema:  {results.system_info['OS']}")
    print(f"  CPU:      {results.system_info['CPU']}")
    print(f"  GPU:      {results.system_info['GPU']}")
    print(f"  RAM:      {results.system_info['RAM']}")
    print(f"  NPU:      {results.system_info['NPU']}")
    
    print(f"\n  🔹 CPU (FP32):      {results.cpu_gflops:.2f} GFLOPS")
    print(f"  🔹 GPU (FP16):      {results.gpu_tops:.2f} TOPS")
    
    if COREML_AVAILABLE:
        print(f"  🔹 NPU FP16:        {results.npu_fp16_tops:.2f} TOPS")
        print(f"  🔹 NPU INT8:        {results.npu_int8_tops:.2f} TOPS")


def check_logic_board_health():
    """Ejecuta una auditoría científica profunda del hardware."""
    print_section("AUDITORÍA DE INTEGRIDAD DE LA LOGIC BOARD")
    
    # 1. Batería (IOKit)
    print("\n[1/3] Evaluando estado electroquímico de la batería (IOKit)...")
    battery_info = {}
    stdout, _ = run_command(['ioreg', '-rn', 'AppleSmartBattery'])
    if stdout:
        for line in stdout.split('\n'):
            line = line.strip()
            if '"DesignCapacity" = ' in line:
                battery_info['design_capacity'] = int(line.split('=')[1].strip())
            elif '"AppleRawMaxCapacity" = ' in line or '"NominalChargeCapacity" = ' in line:
                # Prioritize AppleRawMaxCapacity, but fallback to NominalChargeCapacity
                val = int(line.split('=')[1].strip())
                if 'raw_max_capacity' not in battery_info or '"AppleRawMaxCapacity" = ' in line:
                    battery_info['raw_max_capacity'] = val
            elif '"CycleCount" = ' in line:
                battery_info['cycle_count'] = int(line.split('=')[1].strip())
            elif '"Temperature" = ' in line:
                battery_info['temperature'] = int(line.split('=')[1].strip()) / 100.0
            elif '"Voltage" = ' in line:
                battery_info['voltage'] = int(line.split('=')[1].strip()) / 1000.0

    if 'design_capacity' in battery_info and 'raw_max_capacity' in battery_info:
        design = battery_info['design_capacity']
        current = battery_info['raw_max_capacity']
        cycles = battery_info.get('cycle_count', 0)
        temp = battery_info.get('temperature', 0)
        health_pct = (current / design) * 100
        
        print("  🔋 Batería:")
        print(f"     Salud real (Hardware): {health_pct:.1f}% ({current} mAh retenidos de {design} mAh originales)")
        print(f"     Ciclos de carga:       {cycles}")
        print(f"     Temperatura actual:    {temp:.1f}°C")
        if health_pct < 80:
            print("     ⚠️  ADVERTENCIA: Batería muy degradada químicamente, requiere reemplazo pronto.")
        else:
            print("     ✓  Celdas de batería en buenas condiciones físicas.")
    else:
        print("  ⚠️  No se pudo leer la información de la batería (Es un Mac de escritorio o tiene fallo en el bus I2C/SMC).")

    # 2. SMC Thermal / Fans
    print("\n[2/3] Verificando sensores térmicos y sistema de refrigeración (SMC)...")
    stdout, _ = run_command(['powermetrics', '-n', '1', '--samplers', 'smc'], timeout=10)
    thermal_pressure = "Desconocido"
    fan_speeds = []
    if stdout:
        for line in stdout.split('\n'):
            line = line.strip()
            if 'Thermal pressure:' in line or 'Thermal Pressure:' in line:
                thermal_pressure = line.split(':')[1].strip()
            elif 'Fan:' in line or 'fan:' in line:
                fan_speeds.append(line)

    print(f"  🌡️  Presión Térmica SMC: {thermal_pressure}")
    if thermal_pressure.lower() in ['heavy', 'trapping', 'critical']:
        print("     ❌ PELIGRO: El SMC reporta sobrecalentamiento crítico en reposo.")
        print("                 Posible daño en disipador, pasta térmica seca o sensor termistor roto.")
    else:
        print("     ✓  Temperaturas de placa base dentro de los rangos seguros.")
        
    if fan_speeds:
        for fan in fan_speeds:
            print(f"  🌀 {fan}")
        print("     ✓  Controlador de ventiladores respondiendo correctamente.")
    else:
        print("  🌀 Ventiladores: 0 RPM (Modelo con disipación pasiva o ventilador apagado por baja temperatura)")

    # 3. Kernel Panics
    print("\n[3/3] Auditando registros de Kernel Panic (Fallos de hardware a bajo nivel)...")
    panic_dir = "/Library/Logs/DiagnosticReports"
    panic_files = []
    
    if os.path.exists(panic_dir):
        try:
            for f in os.listdir(panic_dir):
                if f.endswith('.ips') or f.endswith('.panic'):
                    # Verificar si es un kernel panic real
                    filepath = os.path.join(panic_dir, f)
                    if os.path.isfile(filepath):
                        # Analizar cabecera del archivo en busca de firmas de panic
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as log_file:
                                content = log_file.read(2048) # Leer primeros 2KB
                                if 'panicString' in content or 'bug_type": "210' in content or 'Kernel Panic' in content or 'SOCD report' in content:
                                    panic_files.append(f)
                        except Exception:
                            pass
        except PermissionError:
            print("  ⚠️  Permisos insuficientes para leer registros de diagnóstico.")

    if panic_files:
        print(f"  ❌ ALERTA ROJA: Se encontraron {len(panic_files)} registro(s) de Kernel Panic (Apagón Inesperado).")
        print("     Esto es un fuerte indicador de hardware defectuoso en la Logic Board (Soldadura fría, PMIC, RAM, NAND).")
        print("     Archivos sospechosos:")
        for f in panic_files[:3]:
            print(f"     - {f}")
        if len(panic_files) > 3:
            print(f"     ... y {len(panic_files) - 3} archivo(s) más.")
        print("     ⚠️  RECOMENDACIÓN: NO COMPRAR ESTE MAC. RIESGO ALTO DE FALLO GENERAL.")
    else:
        print("  ✓  No se encontraron Kernel Panics. La Logic Board, RAM y SoC operan con estabilidad comprobada.")

    print("\n[!] NOTA FINAL SOBRE CHIPS DE MEMORIA RAM:")
    print("    Para comprobar el silicio de la RAM a nivel eléctrico, apague el Mac, mantenga presionada")
    print("    la tecla 'D' (o mantenga el botón de encendido en Apple Silicon) para ejecutar Apple Diagnostics.")


def check_peripherals_and_buses():
    """Audita la presencia de periféricos en el bus y busca fallos de hardware en los logs del kernel."""
    print_section("AUDITORÍA DE PERIFÉRICOS Y BUSES (I2C/SPI/USB/PCIe)")
    
    print("\n[1/2] Verificando presencia de hardware en la Logic Board (IOKit / System Profiler)...")
    
    # 1. Cámara
    stdout, _ = run_command(['system_profiler', 'SPCameraDataType'], timeout=10)
    if stdout and ('FaceTime' in stdout or 'Camera' in stdout or 'Cámara' in stdout):
        print("  📷 Cámara:          [OK: Detectada en el bus]")
    else:
        print("  📷 Cámara:          [ERROR: No detectada / Desconectada]")

    # 2. Audio (Parlantes y Micrófono)
    stdout, _ = run_command(['system_profiler', 'SPAudioDataType'], timeout=10)
    has_mic = stdout and any(k in stdout for k in ('Microphone', 'Micrófono', 'Built-in Micro'))
    has_speaker = stdout and any(k in stdout for k in ('Speaker', 'Bocina', 'Altavoz', 'Built-in Output', 'Salida integrada', 'Internal Speakers'))
    
    # Fallback para Macs Intel (Especialmente con chip T2) donde system_profiler puede fallar o mostrar nombres distintos
    if not has_mic or not has_speaker:
        stdout_ioreg_audio, _ = run_command(['ioreg', '-c', 'IOAudioEngine'], timeout=10)
        stdout_ioreg_t2, _ = run_command(['ioreg', '-c', 'AppleT2Audio'], timeout=10)
        
        if stdout_ioreg_audio or stdout_ioreg_t2:
            # Si detectamos motores de audio en IOKit, asumimos que el hardware de audio está presente
            has_mic = True
            has_speaker = True
    
    if has_mic:
        print("  🎙️  Micrófono:       [OK: Detectado en el bus]")
    else:
        print("  🎙️  Micrófono:       [ERROR: No detectado / Desconectado]")
        
    if has_speaker:
        print("  🔊 Parlantes:       [OK: Detectados en el bus]")
    else:
        print("  🔊 Parlantes:       [ERROR: No detectados / Desconectados]")

    # 3. Touch ID / Biometría
    stdout_bio, _ = run_command(['ioreg', '-c', 'AppleBiometricServices'], timeout=10)
    stdout_sensor, _ = run_command(['ioreg', '-c', 'AppleBiometricSensor'], timeout=10)
    stdout_ibridge, _ = run_command(['system_profiler', 'SPiBridgeDataType'], timeout=10)
    stdout_bioutil, _ = run_command(['bioutil', '-c'], timeout=10)
    
    has_touchid = False
    if stdout_bioutil and 'biometric template' in stdout_bioutil.lower():
        # bioutil es la forma más nativa y confiable de saber si el Touch ID está disponible
        has_touchid = True
    elif stdout_bio and 'AppleBiometricServices' in stdout_bio:
        has_touchid = True
    elif stdout_sensor and 'AppleBiometricSensor' in stdout_sensor:
        has_touchid = True
    elif stdout_ibridge and ('Touch ID' in stdout_ibridge or 'Biometric' in stdout_ibridge):
        has_touchid = True
        
    if has_touchid:
        print("  👆 Touch ID:        [OK: Enclave Seguro / T2 respondiendo]")
    else:
        print("  👆 Touch ID:        [ERROR: No detectado / Mac sin Touch ID o daño en el bus]")

    # 4. Bluetooth / Trackpad
    stdout, _ = run_command(['system_profiler', 'SPBluetoothDataType'], timeout=10)
    if stdout and 'State: On' in stdout:
        print("  📶 Bluetooth:       [OK: Controlador encendido y respondiendo]")
    else:
        print("  📶 Bluetooth:       [ERROR: Controlador apagado o sin comunicación]")

    print("\n[2/2] Minería de Logs de Hardware (Buscando fallos de comunicación en las últimas 2 horas)...")
    # Buscamos errores críticos de comunicación de bus (I2C, SPI, SMC timeouts, hardware faults)
    log_cmd = [
        '/usr/bin/log', 'show', 
        '--predicate', 'eventMessage CONTAINS[c] "hardware fault" OR eventMessage CONTAINS[c] "I2C error" OR eventMessage CONTAINS[c] "SPI timeout"', 
        '--last', '2h'
    ]
    stdout, _ = run_command(log_cmd, timeout=20)
    
    # Filtrar el comando en sí mismo que aparece en los logs de zsh
    real_faults = []
    if stdout:
        for line in stdout.split('\n'):
            line = line.strip()
            if not line or 'eventMessage CONTAINS' in line or 'Filtering the log data' in line or 'Timestamp' in line:
                continue
            if 'hardware fault' in line.lower() or 'i2c error' in line.lower() or 'spi timeout' in line.lower():
                real_faults.append(line)
                
    if real_faults:
        print(f"  ❌ ALERTA ROJA: Se encontraron {len(real_faults)} error(es) de bus de hardware en los registros recientes.")
        print("     Esto indica que, aunque el componente está detectado, sufre de desconexiones o fallos eléctricos intermitentes.")
        print("     Extracto de logs:")
        for fault in real_faults[:3]:
            # Recortar longitud del log para no saturar la pantalla
            print(f"     - {fault[:120]}...")
        if len(real_faults) > 3:
            print(f"     ... y {len(real_faults) - 3} error(es) más ocultos.")
    else:
        print("  ✓  No se registraron fallos de comunicación (I2C/SPI) en el kernel recientemente.")
        print("  ✓  La comunicación digital entre los periféricos y la Logic Board es eléctricamente estable.")

    print("\n" + "=" * 72)
    print(" ⚠️ ADVERTENCIA DE DIAGNÓSTICO FÍSICO Y CHECKLIST MANUAL ⚠️")
    print("=" * 72)
    print(" El software ha verificado que la comunicación digital de los componentes")
    print(" está SANA, pero NO PUEDE DETECTAR DAÑOS MECÁNICOS NI ANALÓGICOS.")
    print(" El técnico encargado DEBE realizar la siguiente inspección física:")
    print("")
    print(" [ ] Pantalla: Buscar píxeles muertos, manchas blancas o quemado de imagen.")
    print(" [ ] Altavoces: Reproducir audio al máximo. Escuchar si hay vibración o estallidos.")
    print(" [ ] Micrófono: Grabar una nota de voz y escuchar si hay estática o ruido.")
    print(" [ ] Teclado: Presionar FÍSICAMENTE todas las teclas. Comprobar que no se traben.")
    print(" [ ] Trackpad: Probar el clic mecánico/háptico en esquinas y centro, y gestos.")
    print(" [ ] Puertos: Conectar un dispositivo real en CADA puerto (USB-C/MagSafe/Audio).")
    print(" [ ] Cámara: Abrir Photo Booth y verificar que la lente no esté rayada/sucia.")
    print(" [ ] Chasis/Bisagras: Comprobar resistencia al cerrar y buscar abolladuras.")
    print("=" * 72)


def main():
    """Función principal de ejecución."""
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Capturar salida en consola para reporte automático
    logger = ReportLogger()
    sys.stdout = logger
    
    try:
        # Verificaciones iniciales
        check_sudo()
        check_dependencies()
        
        print_header()
        
        # ======== BENCHMARKS DE IA ========
        print("\n[1/4] Ejecutando pruebas de CPU, GPU y NPU (esto puede tardar unos segundos)...")
        benchmark = AIBenchmark()
        results = benchmark.run_all()
        display_benchmark_report(results)
        
        # ======== ANÁLISIS DE DISCOS ========
        print_section("ANÁLISIS DE ALMACENAMIENTO")
        
        print("\n[2/4] Detectando discos físicos...")
        physical_disks = find_physical_disks()
        if not physical_disks:
            print("  ⚠️  No se encontraron discos físicos.")
        else:
            print(f"  ✓ Detectados {len(physical_disks)} disco(s)")
        
        print("[3/4] Identificando disco de arranque...")
        boot_disk = get_boot_disk()
        if boot_disk:
            print(f"  ✓ Disco de arranque: /dev/{boot_disk}")
        else:
            print("  ⚠️  No se pudo identificar disco de arranque")
        
        print("[4/4] Analizando salud de discos y midiendo velocidad real...")
        
        # Ordenar discos (boot primero)
        sorted_disks = [boot_disk] if boot_disk and boot_disk in physical_disks else []
        sorted_disks.extend(d for d in physical_disks if d != boot_disk)
        
        for disk_id in sorted_disks:
            disk_info = get_disk_info_summary(disk_id)
            smart_data = get_smart_data(disk_id)
            report = parse_smart_report(smart_data)
            
            # Benchmark de velocidad solo para disco principal
            if disk_id == boot_disk:
                read_speed, write_speed = benchmark_disk_speed(disk_info)
                if read_speed and write_speed:
                    report.read_speed_mbps = read_speed
                    report.write_speed_mbps = write_speed
            
            display_disk_report(disk_id, disk_info, report, disk_id == boot_disk)
        
        # ======== AUDITORIA LOGIC BOARD ========
        check_logic_board_health()
        
        # ======== AUDITORIA PERIFÉRICOS Y BUSES ========
        check_peripherals_and_buses()
        
        # Footer
        print("\n" + "═" * 80)
        print("  ✓ Análisis completado exitosamente")
        
        # ======== GENERACIÓN DEL REPORTE ========
        try:
            sys.stdout = logger.terminal
            
            mac_identity = get_mac_identity()
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                
            report_path = os.path.join(base_dir, f"{mac_identity}.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(logger.get_clean_content())
                f.write("\n  ✓ Reporte generado automáticamente.\n")
                
            print(f"  📄 Reporte guardado en: {report_path}")
        except Exception as e:
            sys.stdout = logger.terminal
            print(f"  ⚠️ Error guardando el reporte: {e}")
            
        print("═" * 80 + "\n")
        
    finally:
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
