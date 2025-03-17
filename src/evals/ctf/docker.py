"""
Docker configuration for CTF tasks.
"""

from pathlib import Path

DEFAULT_APT_GET_INSTALLS = [
    "bash",
    "python3",
    "python3-pip",
    "psmisc",
    "bsdmainutils",
    "cron",
    "imagemagick",
    "tcpdump",
    "dnsutils",
    "git",
    "wget",
    "gcc",
    "sleuthkit",
    "netcat-traditional",
    "unzip",
    "curl",
    "gdb",
    "poppler-utils",
    "tmux",
    "jq",
]

DEFAULT_PIP3_INSTALLS = [
    "pyshark",
    "scapy",
    "pwntools",
    "dpkt",
    "pycryptodome",
    "cryptography",
    "owiener",
    "numpy",
    "sympy",
    "requests",
    "Pillow",
    "python-whois",
    "PyPDF2",
    "flake8",
]


def generate_dockerfile(apt_get_installs: list[str], pip3_installs: list[str]) -> None:
    """Generate a Dockerfile for CTF tasks.

    Args:
        apt_get_installs: List of apt-get packages to install
        pip3_installs: List of pip3 packages to install
    """
    current_dir = Path.cwd()
    template_path = current_dir / "Dockerfile.template"
    dockerfile_path = current_dir / "Dockerfile"

    with open(template_path, "r") as template_file:
        template_content = template_file.read()

    dockerfile_content = template_content.format(
        apt_get_installs=" ".join(apt_get_installs),
        pip3_installs=" ".join(pip3_installs),
    )

    with open(dockerfile_path, "w") as dockerfile:
        dockerfile.write(dockerfile_content)
