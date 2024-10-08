#!/usr/bin/env python3

import rospy

from sensor_msgs.msg import BatteryState


# # Constants are chosen to match the enums in the linux kernel
# # defined in include/linux/power_supply.h as of version 3.7
# # The one difference is for style reasons the constants are
# # all uppercase not mixed case.

# # Power supply status constants
# uint8 POWER_SUPPLY_STATUS_UNKNOWN = 0
# uint8 POWER_SUPPLY_STATUS_CHARGING = 1
# uint8 POWER_SUPPLY_STATUS_DISCHARGING = 2
# uint8 POWER_SUPPLY_STATUS_NOT_CHARGING = 3
# uint8 POWER_SUPPLY_STATUS_FULL = 4

# # Power supply health constants
# uint8 POWER_SUPPLY_HEALTH_UNKNOWN = 0
# uint8 POWER_SUPPLY_HEALTH_GOOD = 1
# uint8 POWER_SUPPLY_HEALTH_OVERHEAT = 2
# uint8 POWER_SUPPLY_HEALTH_DEAD = 3
# uint8 POWER_SUPPLY_HEALTH_OVERVOLTAGE = 4
# uint8 POWER_SUPPLY_HEALTH_UNSPEC_FAILURE = 5
# uint8 POWER_SUPPLY_HEALTH_COLD = 6
# uint8 POWER_SUPPLY_HEALTH_WATCHDOG_TIMER_EXPIRE = 7
# uint8 POWER_SUPPLY_HEALTH_SAFETY_TIMER_EXPIRE = 8

# # Power supply technology (chemistry) constants
# uint8 POWER_SUPPLY_TECHNOLOGY_UNKNOWN = 0
# uint8 POWER_SUPPLY_TECHNOLOGY_NIMH = 1
# uint8 POWER_SUPPLY_TECHNOLOGY_LION = 2
# uint8 POWER_SUPPLY_TECHNOLOGY_LIPO = 3
# uint8 POWER_SUPPLY_TECHNOLOGY_LIFE = 4
# uint8 POWER_SUPPLY_TECHNOLOGY_NICD = 5
# uint8 POWER_SUPPLY_TECHNOLOGY_LIMN = 6

# Header  header
# float32 voltage          # Voltage in Volts (Mandatory)
# float32 current          # Negative when discharging (A)  (If unmeasured NaN)
# float32 charge           # Current charge in Ah  (If unmeasured NaN)
# float32 capacity         # Capacity in Ah (last full capacity)  (If unmeasured NaN)
# float32 design_capacity  # Capacity in Ah (design capacity)  (If unmeasured NaN)
# float32 percentage       # Charge percentage on 0 to 1 range  (If unmeasured NaN)
# uint8   power_supply_status     # The charging status as reported. Values defined above
# uint8   power_supply_health     # The battery health metric. Values defined above
# uint8   power_supply_technology # The battery chemistry. Values defined above
# bool    present          # True if the battery is present

# float32[] cell_voltage   # An array of individual cell voltages for each cell in the pack
#                          # If individual voltages unknown but number of cells known set each to NaN
# string location          # The location into which the battery is inserted. (slot number or plug)
# string serial_number     # The best approximation of the battery serial number


class FakeBattery:
    def __init__(self):
        self.robot_frame = rospy.get_param("~robot_frame", "base_footprint")
        self.update_frequency = rospy.get_param("~update_frequency", 1.0)
        self.fake_time_used = rospy.get_param("~fake_time_used", 8.0)
        self.rate = rospy.Rate(self.update_frequency)

        self.battery_pub = rospy.Publisher("/battery_state", BatteryState, queue_size=5)

        self.msg = BatteryState()
        self.msg.header.frame_id = self.robot_frame
        self.msg.voltage = 24.0
        self.msg.current = -6.65
        self.msg.charge = 78.0
        self.msg.capacity = 78.0
        self.msg.design_capacity = 80.0
        self.msg.percentage = self.msg.charge / self.msg.capacity
        self.msg.power_supply_status = 0
        self.msg.power_supply_health = 0
        self.msg.power_supply_technology = 4
        self.msg.present = True
        self.msg.cell_voltage = [3.31, 3.32, 3.33, 3.34, 3.31, 3.32, 3.33, 3.34]
        self.msg.cell_temperature = [33.4, 30.89]

        # vars:
        self.capity_used_per_rate = self.msg.capacity / (
            self.update_frequency * self.fake_time_used * 3600
        )

    def run(self):
        while not rospy.is_shutdown():
            self.msg.charge -= self.capity_used_per_rate
            self.msg.percentage = self.msg.charge / self.msg.capacity
            self.msg.header.stamp = rospy.Time.now()
            self.battery_pub.publish(self.msg)
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("fake_battery")
    try:
        fake_battery = FakeBattery()
        rospy.logwarn("Follow waypoints server node is running!")
        fake_battery.run()
    except rospy.ROSInterruptException:
        pass
