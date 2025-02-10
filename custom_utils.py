import requests
from colorama import Fore, Style
import time
import torch
import os
import subprocess
import sys

def log(msg, emoji="✔️", error=False):
    if error:
        emoji="❌"
        print(f"{Fore.RED}{emoji} {msg}{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}{emoji} {msg}{Style.RESET_ALL}")

def print_gpu():
    if torch.cuda.is_available():
        log(f"CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'} | Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    else:
        log("CUDA: False | No GPU available", error=True)

    
def check_server(url, server_name, retries=50, delay=500):
    for i in range(retries):
        try:
            response = requests.get(url)

            if response.status_code == 200:
                log(f"{server_name} is reachable on {url}")
                return True
        except requests.RequestException as e:
            pass

        time.sleep(delay / 1000)

    log(
        f"Failed to connect to {server_name} at {url} after {retries} attempts.", error=True
    )
    return False

def get_public_ip(version='ipv4'):
    try:
        url = f'https://api64.ipify.org?format=json&{version}=true'
        response = requests.get(url)
        data = response.json()
        public_ip = data['ip']
        return public_ip
    except Exception as e:
        log(f"Error getting public {version} address:", error=True)
        


def install_custom_node_requirements(base_dir):
    for item in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, item)
        
        if os.path.isdir(dir_path):
            req_file = os.path.join(dir_path, 'requirements.txt')
            if os.path.exists(req_file):
                try:
                    log(f"Installing requirements: {dir_path}")
                    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], 
                                check=True,
                                cwd=dir_path)
                except subprocess.CalledProcessError as e:
                    log(f"Error installing requirements in {dir_path}: {e}", error=True)