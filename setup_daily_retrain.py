"""
Set up Windows Task Scheduler to run model retraining daily at 2:00 AM.
Run this script once: python setup_daily_retrain.py
"""
import os
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(PROJECT_DIR, "train_models.py")
TASK_NAME = "PharmacyDashboard_DailyModelRetrain"


def setup_task():
    """Create a scheduled task to run retraining at 2am daily."""
    if not os.path.exists(TRAIN_SCRIPT):
        print(f"Error: {TRAIN_SCRIPT} not found")
        return False

    # Use cmd to change directory and run Python (ensures correct cwd for CSV/DB paths)
    action = f'cmd /c "cd /d {PROJECT_DIR} && python train_models.py"'
    
    cmd = [
        "schtasks", "/create",
        "/tn", TASK_NAME,
        "/tr", action,
        "/sc", "daily",
        "/st", "02:00",
        "/f"  # overwrite if exists
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("✅ Scheduled task created successfully!")
            print(f"   Task '{TASK_NAME}' will run daily at 2:00 AM.")
            print(f"   To run manually: schtasks /run /tn \"{TASK_NAME}\"")
            print(f"   To remove: schtasks /delete /tn \"{TASK_NAME}\" /f")
            return True
        else:
            print("Error creating task:", result.stderr or result.stdout)
            return False
    except Exception as e:
        print(f"Error: {e}")
        print("\nManual setup: Open Task Scheduler, create a new task:")
        print(f"  - Trigger: Daily at 2:00 AM")
        print(f"  - Action: Start program = {BAT_PATH}")
        return False


def remove_task():
    """Remove the scheduled task."""
    result = subprocess.run(
        ["schtasks", "/delete", "/tn", TASK_NAME, "/f"],
        capture_output=True, text=True, shell=True
    )
    if result.returncode == 0:
        print(f"✅ Task '{TASK_NAME}' removed.")
    else:
        print("Task may not exist or could not be removed.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "remove":
        remove_task()
    else:
        setup_task()
