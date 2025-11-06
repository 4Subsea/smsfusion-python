class ConingTrajectorySimulator:
    """
    Coning trajectory generator and IMU simulator.

    A coning trajectory is defined as a circular motion of a vector, r, of constant
    amplitude, making a constant angle, theta, with respect to a fixed axis. Here,
    the fixed axis is the z-axis.

    Let,
    - R be the amplitude of the vector, r.
    - theta be the coning (half) angle. I.e., the angle between r and the z-axis.
    - phi be the heading angle. I.e., the angle between the projection of r onto the
      x-y plane and the x-axis.
    
    Then, the vector, r, can be expressed as:

        r(t) = R * [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]^T

    """


    def __init__(self):
        pass

    def __call__(self, n: int):
        pass