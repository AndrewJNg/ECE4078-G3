import numpy as np
# Pulse width 500-2400 micro s
# Pulse cycle ca. 20ms


# deg  pulse
#  0   1.5ms, 
#  90  2  ms,
# -90  1  ms,

"""Simple test for a standard servo on channel 0 and a continuous rotation servo on channel 1."""
"""
import time
from adafruit_servokit import ServoKit

# Set channels to the number of servo channels on your kit.
# 8 for FeatherWing, 16 for Shield/HAT/Bonnet.
kit = ServoKit(channels=8)

kit.servo[0].angle = 180
time.sleep(2)
kit.servo[0].angle = 0
"""

class Servo():
    def __init__(self, min_pulse, max_pulse):
        self.pulse = 0
        self.min_pulse = min_pulse
        self.max_pulse = max_pulse
    
    def angleToPulse(self, angle):
        self.pulse = np.interp(angle, [self.min_pulse, self.max_pulse, [0, 180]]) # pulse width
        
    # TODO
    def runServo(self):
        # Check if value is between 0 to 180

        # Run function to convert to 0 to 180

        # Convert angle to pulse width

        # Update self.pulse

        # Actuate servo
        pass

def callibrate(servo, min_pulse, max_pulse):
    servo.min_pulse = min_pulse
    servo.max_pulse = max_pulse

def intInput(msg):
    while True:
        try:
            value = input(msg)
            break
        except ValueError:
            continue
    return value

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.209')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    args, _ = parser.parse_known_args()

    # Initialize servo class
    servo = Servo(min_pulse=1000, max_pulse=2000)
    
    print('Running servo callibration')
    while True:
        # Resetting servo
        print ('Reseting to initial position')
        servo.runServo(0)
        # Actuate servo
        servo.runServo(180)
        # Prompt user input and update pulse width values
        uInput = input('Did servo turn correctly? [y/n]')
        if uInput == 'n' or uInput == 'N':
            print(f'\nLast min_pulse: {servo.min_pulse}us\nLast max_pulse: {servo.min_pulse}us')
            min_pulse = intInput('\nEnter new min_pulse: ')
            max_pulse = intInput('Enter new min_pulse: ')
            # Update to new pulse widths
            callibrate(servo, min_pulse, max_pulse)
        elif uInput == 'y' or uInput == 'Y':
            print(f'\nCallibrated:\nmin_pulse: {servo.min_pulse}us\nmax_pulse: {servo.min_pulse}us')
        else:
            print('ERROR: Invalid input. Try again.')
            continue