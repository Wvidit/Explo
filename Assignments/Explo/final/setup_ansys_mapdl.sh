#!/usr/bin/env bash
###############################################################################
#  setup_ansys_mapdl.sh
#  ---------------------------------------------------------------------------
#  Installs only the relevant PyAnsys packages for Ansys MAPDL using the
#  active pyenv global Python (no virtual environment is created).
#  Does NOT download the full Ansys software (requires a licensed installer
#  from customer.ansys.com).
#
#  Usage:
#    chmod +x setup_ansys_mapdl.sh
#    ./setup_ansys_mapdl.sh                  # full setup (needs sudo for deps)
#    ./setup_ansys_mapdl.sh --python-only    # skip system packages
#
#  Environment overrides (optional):
#    ANSYS_DIR=/ansys_inc/v241  LICENSE_SERVER=1055@myserver  ./setup_ansys_mapdl.sh
###############################################################################
set -euo pipefail

# ─── Config ─────────────────────────────────────────────────────────────────
ANSYS_DIR="${ANSYS_DIR:-}"                    # e.g. /home/user/ansys_inc/v241
LICENSE_SERVER="${LICENSE_SERVER:-1055@localhost}"
SHELL_RC="$HOME/.bashrc"
PYTHON_ONLY=false

# PyAnsys packages relevant to MAPDL (pip-installable, no Ansys license needed)
PYANSYS_PACKAGES=(
  "ansys-mapdl-core"       # Core PyMAPDL client
  "ansys-mapdl-reader"     # Read MAPDL result files (.rst, .cdb, etc.)
  "ansys-dpf-core"         # Data Processing Framework – post-processing
  "ansys-dpf-post"         # Higher-level DPF post-processing API
  "ansys-platform-instancemanagement"  # Manage remote MAPDL instances
)

# ─── Parse flags ────────────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --python-only) PYTHON_ONLY=true ;;
    --help|-h)
      grep '^#' "$0" | head -14 | tail -12 | sed 's/^# \?//'
      exit 0 ;;
  esac
done

# ─── Resolve Python/pip from pyenv ──────────────────────────────────────────
if command -v pyenv &>/dev/null; then
  eval "$(pyenv init -)"
  PYTHON_CMD="$(pyenv which python)"
  PIP_CMD="$(pyenv which pip)"
else
  PYTHON_CMD="python3"
  PIP_CMD="pip3"
fi

# ─── Helpers ────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${GREEN}══ $* ══${NC}"; }

###############################################################################
# STEP 1 – System dependencies (needed to run the MAPDL solver when present)
###############################################################################
install_sys_deps() {
  step "System Dependencies"
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO_ID="${ID:-unknown}"
  else
    error "Cannot detect distro (/etc/os-release missing)."
  fi

  case "$DISTRO_ID" in
    ubuntu|debian|linuxmint|pop)
      info "Detected: $DISTRO_ID — installing apt packages …"
      sudo apt-get update -qq
      sudo apt-get install -y -qq \
        libxcrypt-compat libnsl2 libglu1-mesa libxrender1 \
        libxcursor1 libxft2 libxinerama1 libxi6 libxtst6 \
        python3 python3-venv python3-pip 2>/dev/null
      ;;
    rhel|centos|fedora|rocky|almalinux)
      info "Detected: $DISTRO_ID — installing yum packages …"
      sudo yum install -y -q \
        mesa-libGLU libXrender libXcursor libXft libXinerama \
        libXi libXtst python3 python3-pip 2>/dev/null
      ;;
    *)
      warn "Unsupported distro '$DISTRO_ID'. Skipping OS package install."
      warn "Manually install: libGLU, libXrender, libXcursor, python3, python3-venv"
      ;;
  esac
  info "System dependencies done."
}

###############################################################################
# STEP 2 – Confirm pyenv Python
###############################################################################
check_python() {
  step "Python Environment (pyenv global)"
  info "Python : $($PYTHON_CMD --version)"
  info "pip    : $($PIP_CMD --version)"
  info "Executable: $PYTHON_CMD"
  $PIP_CMD install --upgrade pip --quiet
}

###############################################################################
# STEP 3 – Install PyAnsys packages into pyenv global
###############################################################################
install_pyansys_packages() {
  step "PyAnsys Package Installation"
  for pkg in "${PYANSYS_PACKAGES[@]}"; do
    info "Installing $pkg …"
    $PIP_CMD install "$pkg" --quiet || warn "Failed to install $pkg (may not exist for your Python version)"
  done
  info "PyAnsys packages installed."
}

###############################################################################
# STEP 4 – Configure shell environment
###############################################################################
configure_env() {
  step "Shell Environment"
  local MARKER="# >>> ANSYS MAPDL CONFIG >>>"
  if grep -q "$MARKER" "$SHELL_RC" 2>/dev/null; then
    info "Ansys env block already in $SHELL_RC — skipping."
    return 0
  fi

  {
    echo ""
    echo "$MARKER"
    echo "export ANSYSLMD_LICENSE_FILE=\"$LICENSE_SERVER\""
    if [ -n "$ANSYS_DIR" ]; then
      echo "export ANSYS_DIR=\"$ANSYS_DIR\""
      echo "export PATH=\"\$ANSYS_DIR/ansys/bin:\$PATH\""
    fi
    echo "# <<< ANSYS MAPDL CONFIG <<<"
  } >> "$SHELL_RC"
  info "Environment written to $SHELL_RC."
}

###############################################################################
# STEP 5 – Write verification script
###############################################################################
write_verify_script() {
  step "Verification Script"
  local VERIFY="$HOME/verify_mapdl.py"
  cat > "$VERIFY" <<'PYEOF'
#!/usr/bin/env python3
"""Verify PyAnsys packages are importable and MAPDL can (optionally) launch."""
import importlib, sys, os

packages = {
    "ansys.mapdl.core":   "ansys-mapdl-core",
    "ansys.mapdl.reader": "ansys-mapdl-reader",
    "ansys.dpf.core":     "ansys-dpf-core",
    "ansys.dpf.post":     "ansys-dpf-post",
}

all_ok = True
for module, pkg in packages.items():
    try:
        importlib.import_module(module)
        print(f"  [OK]      {pkg}")
    except ImportError:
        print(f"  [MISSING] {pkg}  →  pip install {pkg}")
        all_ok = False

ansys_dir = os.environ.get("ANSYS_DIR", "")
if ansys_dir:
    print(f"\nANSYS_DIR = {ansys_dir}")
    try:
        from ansys.mapdl.core import launch_mapdl
        mapdl = launch_mapdl()
        print(f"  [OK]  MAPDL launched: {mapdl.version}")
        mapdl.exit()
    except Exception as e:
        print(f"  [FAIL] {e}\n        (Is ANSYS installed and licensed?)")
else:
    print("\n[INFO] ANSYS_DIR not set — skipping solver launch check.")

sys.exit(0 if all_ok else 1)
PYEOF
  chmod +x "$VERIFY"
  info "Verification script written: $VERIFY"
}

###############################################################################
# STEP 6 – Summary
###############################################################################
print_summary() {
  local INSTALLED_PKGS
  INSTALLED_PKGS=$($PIP_CMD list 2>/dev/null | grep -i ansys || echo "(none)")
  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║          PyAnsys MAPDL Setup — Done                        ║"
  echo "╠══════════════════════════════════════════════════════════════╣"
  echo "  Python    : $PYTHON_CMD ($($PYTHON_CMD --version))"
  echo "  Verify    : $PYTHON_CMD $HOME/verify_mapdl.py"
  echo "  RC file   : $SHELL_RC"
  echo ""
  echo "  Installed PyAnsys packages:"
  echo "$INSTALLED_PKGS" | sed 's/^/    /'
  echo "╠══════════════════════════════════════════════════════════════╣"
  echo "  NOTE: The MAPDL solver itself requires a licensed installer"
  echo "  from https://customer.ansys.com (not installed by this script)"
  echo "╚══════════════════════════════════════════════════════════════╝"
}

###############################################################################
# Main
###############################################################################
info "═══ PyAnsys / MAPDL Package Setup (pyenv global) ═══"

if ! $PYTHON_ONLY; then
  install_sys_deps
fi

check_python
install_pyansys_packages
configure_env
write_verify_script
print_summary
