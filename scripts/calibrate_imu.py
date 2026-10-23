"""Calibrate the BNO08x and save DCD in sensor memory (no BNO055 pickle).

Keep the robot still briefly, then slowly rotate it through all three axes.
"""
from mini_bdx_runtime.raw_imu import Imu

if __name__ == "__main__":
    Imu(50, calibrate=True, upside_down=False)
    print("BNO08x calibration saved in sensor")
