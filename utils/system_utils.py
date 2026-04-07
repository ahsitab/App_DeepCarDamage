import psutil
import platform
import os
import time

def get_system_metrics():
    """
    Get current system hardware performance metrics.
    """
    try:
        # CPU Info
        cpu_usage = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count(logical=True)
        
        # Memory Info
        memory = psutil.virtual_memory()
        ram_total = memory.total / (1024 ** 3)  # GB
        ram_used = memory.used / (1024 ** 3)   # GB
        ram_percent = memory.percent
        
        # OS Info
        os_info = f"{platform.system()} {platform.release()}"
        python_version = platform.python_version()
        
        return {
            'cpu_usage': cpu_usage,
            'cpu_count': cpu_count,
            'ram_total': round(ram_total, 2),
            'ram_used': round(ram_used, 2),
            'ram_percent': ram_percent,
            'os': os_info,
            'python': python_version,
            'timestamp': time.time()
        }
    except Exception as e:
        return {'error': str(e)}

def get_model_file_info(models_dir, filenames):
    """
    Get file sizes and metadata for a list of model files.
    """
    results = []
    for fname in filenames:
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            results.append({
                'name': fname,
                'path': path,
                'size_mb': round(size_mb, 2),
                'status': 'Online'
            })
        else:
            results.append({
                'name': fname,
                'path': path,
                'size_mb': 0,
                'status': 'Offline'
            })
    return results
