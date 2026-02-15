#!/usr/bin/env python3
"""
macOS Hardware Info - Análisis completo de hardware para macOS.

Consolida análisis de SSD (salud, temperatura, TBW) y benchmarks de IA (CPU, GPU, NPU).
Compatible con SSDs estándar y Apple Silicon. Requiere Python 3.13+ y privilegios sudo.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
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
def check_dependencies() -> bool:
    """Verifica dependencias del sistema."""
    if not shutil.which("smartctl"):
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
    """Identifica el disco de arranque del sistema."""
    stdout, _ = run_command(['diskutil', 'info', '/'])
    if not stdout:
        return None
    
    for line in stdout.split('\n'):
        if 'Part of Whole:' in line:
            if match := DISK_ID_PATTERN.search(line):
                return match.group(1)
    return None


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
    
    # Intentar múltiples métodos
    for cmd in [
        ['smartctl', '-a', '-j', device_path],
        ['smartctl', '-d', 'auto', '-T', 'permissive', '-a', '-j', device_path],
        ['smartctl', '-x', '-j', device_path]
    ]:
        if result := run_command_json(cmd):
            return result
    
    return None


def benchmark_disk_speed(mount_point: str = '/tmp') -> tuple[float | None, float | None]:
    """Benchmark de velocidad real del disco (lectura/escritura)."""
    TEST_SIZE_MB = 1024
    BLOCK_SIZE = 1048576
    TOTAL_BLOCKS = TEST_SIZE_MB
    
    try:
        test_dir = Path(mount_point) / '.DiskSpeedTest'
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / 'testfile.dat'
        
        block_data = os.urandom(BLOCK_SIZE)
        
        # Test de escritura
        start_write = time.perf_counter()
        with open(test_file, 'wb', buffering=0) as f:
            for _ in range(TOTAL_BLOCKS):
                f.write(block_data)
            f.flush()
            os.fsync(f.fileno())
        write_time = time.perf_counter() - start_write
        write_speed = TEST_SIZE_MB / write_time if write_time > 0 else None
        
        # Purgar caché
        subprocess.run(['purge'], capture_output=True, timeout=60, check=False)
        time.sleep(2)
        
        # Test de lectura
        start_read = time.perf_counter()
        with open(test_file, 'rb') as f:
            while f.read(BLOCK_SIZE):
                pass
        read_time = time.perf_counter() - start_read
        read_speed = TEST_SIZE_MB / read_time if read_time > 0 else None
        
        # Limpieza
        test_file.unlink(missing_ok=True)
        test_dir.rmdir()
        
        return read_speed, write_speed
    except Exception:
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


def main():
    """Función principal de ejecución."""
    # Verificaciones iniciales
    check_sudo()
    check_dependencies()
    
    print_header()
    
    # ======== ANÁLISIS DE DISCOS ========
    print_section("ANÁLISIS DE ALMACENAMIENTO")
    
    print("\n[1/3] Detectando discos físicos...")
    physical_disks = find_physical_disks()
    if not physical_disks:
        print("  ⚠️  No se encontraron discos físicos.")
    else:
        print(f"  ✓ Detectados {len(physical_disks)} disco(s)")
    
    print("[2/3] Identificando disco de arranque...")
    boot_disk = get_boot_disk()
    if boot_disk:
        print(f"  ✓ Disco de arranque: /dev/{boot_disk}")
    else:
        print("  ⚠️  No se pudo identificar disco de arranque")
    
    print("[3/3] Analizando salud de discos...")
    
    # Ordenar discos (boot primero)
    sorted_disks = [boot_disk] if boot_disk and boot_disk in physical_disks else []
    sorted_disks.extend(d for d in physical_disks if d != boot_disk)
    
    for disk_id in sorted_disks:
        disk_info = get_disk_info_summary(disk_id)
        smart_data = get_smart_data(disk_id)
        report = parse_smart_report(smart_data)
        
        # Benchmark de velocidad solo para disco principal
        if disk_id == boot_disk:
            read_speed, write_speed = benchmark_disk_speed()
            if read_speed and write_speed:
                report.read_speed_mbps = read_speed
                report.write_speed_mbps = write_speed
        
        display_disk_report(disk_id, disk_info, report, disk_id == boot_disk)
    
    # ======== BENCHMARKS DE IA ========
    print("\n")
    benchmark = AIBenchmark()
    results = benchmark.run_all()
    display_benchmark_report(results)
    
    # Footer
    print("\n" + "═" * 80)
    print("  ✓ Análisis completado exitosamente")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    main()
