import datetime
import time

def display_welcome():
    """Prints a nice welcome message and the current system time."""
    
    # Get current time formatted nicely
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    welcome_message = f"""
======================================================
       ✨ Hello World! Welcome to the Agent Workspace ✨
======================================================
We have successfully initialized the Python environment!
System time when script ran: {current_time}
======================================================
"""
    print(welcome_message)

if __name__ == "__main__":
    display_welcome()