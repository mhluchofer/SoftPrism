import numpy as np
import pandas as pd

from model.system_motors import SystemMotors
from model.sensor import Sensor

from softprism import Geometry, Material, ActuationConfig, SoftPrism, TendonMotorBridge


# ------------------------------------------------------------------
# SoftPrism model + tendon-motor bridge
# ------------------------------------------------------------------
# Replaces the old InverseKinematics class: motor_angles_from_q(q)
# computes theta1, theta2, theta3 using the verified PCC tendon-length
# model (Sec. II.D), instead of the previous ad hoc formula.
geometry = Geometry(
    length=0.105,
    n_prisms=5,
    prism_side=0.07,
    prism_thickness=0.014,
    backbone_radius=0.01
)

material = Material(
    backbone_mass=0.03,
    prism_mass=0.01,
    young_modulus=2e6
)

actuationconfig = ActuationConfig(
    n_tendons=3,
    tendon_distance=0.005
)

robot = SoftPrism(geometry, material, actuationconfig)

# pulley_radius: confirmed hardware, 2 cm diameter, direct pulley (no gearbox)
bridge = TendonMotorBridge(robot, pulley_radius=0.01)


# ------------------------------------------------------------------
# Motors
# ------------------------------------------------------------------
motors = SystemMotors(3)  # instantiate SystemMotors class >> number of motors
motors.loadMotors([1, 2, 3], "SoftNeckMotorConfig.json")  # motor's ids
motors.startMotors()  # start motors

# ------------------------------------------------------------------
# Sensor
# ------------------------------------------------------------------
mi_sensor = Sensor()  # instantiate Sensor class
mi_sensor.sensorStream()  # enable sensor

# Parameters of the DataFrame
cols = ['Inclination', 'Orientation', 'M1', 'M2', 'M3']
data = []

# ------------------------------------------------------------------
# Sweep: inclination (deg) x orientation (deg)
# ------------------------------------------------------------------
for inclination in range(5, 36, 5):
    for orientation in range(5, 361, 10):

        # NOTE: inclination/orientation are in degrees (matching the
        # original sweep ranges); softprism's q = [theta, phi] is in
        # radians (Sec. II.A), so both are converted here.
        theta = np.radians(inclination)
        phi = np.radians(orientation)

        theta1, theta2, theta3 = bridge.motor_angles_from_q([theta, phi])

        motors.setupPositionsMode(12, 12)  # setting velocity and acceleration values
        motors.setPositions([theta1, theta2, theta3])

        # Knowing the Inclination and Orientation of the sensor, with a previous motor position
        for i in np.arange(0, 2, 0.02):  # time sampling >> steps of 0.02
            incli, orient = mi_sensor.readSensorNeck(mi_sensor)

            print("Inclination: ", round(incli, 1),
                  " Orientation: ", round(orient, 1))

        # Adding the values of incli, orient and encoders in "data"
        data.append([incli, orient, motors.motorsArray[0].getPosition(
        ), motors.motorsArray[1].getPosition(), motors.motorsArray[2].getPosition()])

    # adding the data values (array type), to the data frame
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(
        '/home/sofia/Documents/03072026_neck_carne_pinn_train.csv', index=False)
    df.info()

    print("Inclination: ", round(incli, 1), " Orientation: ", round(orient, 1))

print("Data Ready")

theta1, theta2, theta3 = bridge.motor_angles_from_q([theta, phi])
motors.setPositions([theta1, theta2, theta3])
