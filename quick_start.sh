#!/bin/bash
# Quick Start Script for Autonomous Landing Vision System
# This script automates the installation and setup process

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   Autonomous Drone Landing Vision System - Quick Start   ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "⚠ Warning: Not running on Raspberry Pi"
    echo "Some features may not work properly"
    echo ""
fi

# Update system
echo "📦 Step 1: Updating system packages..."
sudo apt update
echo "✓ System updated"
echo ""

# Install system dependencies
echo "📦 Step 2: Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-opencv \
    libatlas-base-dev \
    python3-venv \
    v4l-utils
echo "✓ Dependencies installed"
echo ""

# Enable camera if on Raspberry Pi
if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "📷 Step 3: Enabling camera..."
    if ! grep -q "start_x=1" /boot/config.txt; then
        echo "start_x=1" | sudo tee -a /boot/config.txt
        echo "gpu_mem=128" | sudo tee -a /boot/config.txt
    fi
    echo "✓ Camera enabled (requires reboot)"
    echo ""
fi

# Enable UART for MAVLink
if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "📡 Step 4: Configuring UART for MAVLink..."
    if ! grep -q "enable_uart=1" /boot/config.txt; then
        echo "enable_uart=1" | sudo tee -a /boot/config.txt
        echo "dtoverlay=disable-bt" | sudo tee -a /boot/config.txt
    fi
    
    # Disable serial console
    sudo systemctl stop serial-getty@ttyAMA0.service
    sudo systemctl disable serial-getty@ttyAMA0.service
    
    echo "✓ UART configured"
    echo ""
fi

# Create virtual environment
echo "🐍 Step 5: Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
echo "✓ Virtual environment created"
echo ""

# Install Python packages
echo "📦 Step 6: Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Python packages installed"
echo ""

# Test camera
echo "📷 Step 7: Testing camera..."
if v4l2-ctl --list-devices > /dev/null 2>&1; then
    echo "✓ Camera detected"
    v4l2-ctl --list-devices
else
    echo "⚠ No camera detected"
    echo "Please connect a camera and reboot"
fi
echo ""

# Run tests
echo "🧪 Step 8: Running system tests..."
python3 test_vision_system.py
TEST_RESULT=$?
echo ""

# Final instructions
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE                         ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo "✓ All tests passed!"
else
    echo "⚠ Some tests failed. Please review the output above."
fi

echo ""
echo "📖 Quick Start Commands:"
echo ""
echo "  # Basic test (no MAVLink)"
echo "  python3 landing_vision_system.py"
echo ""
echo "  # With MAVLink (SpeedyBee F4 V3)"
echo "  python3 landing_vision_system.py --mavlink --connection /dev/ttyAMA0"
echo ""
echo "  # Test mode (30 seconds)"
echo "  python3 landing_vision_system.py --duration 30"
echo ""
echo "📚 Documentation: See README.md for detailed instructions"
echo ""

# Check if reboot needed
if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    if ! grep -q "start_x=1" /boot/config.txt.bak 2>/dev/null; then
        echo "⚠ REBOOT REQUIRED to enable camera and UART"
        echo ""
        read -p "Reboot now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo reboot
        fi
    fi
fi

echo "🚁 Happy flying!"
echo ""
