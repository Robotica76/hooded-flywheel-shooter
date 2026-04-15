import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# -------------------------
# Utilities
# -------------------------
def rpm_to_radps(rpm: float) -> float:
    return rpm * 2*np.pi / 60.0

def radps_to_rpm(radps: float) -> float:
    return radps * 60.0 / (2*np.pi)

# -------------------------
# Flywheel shooter model (from FlywheelShooter notebook)
# -------------------------
@dataclass
class FlywheelShooterParams:
    # Projectile
    m_p: float     # kg
    r_p: float     # m
    k_p: float     # shape factor (sphere=2/5, etc.)

    # Wheel
    m_w: float     # kg
    r_w: float     # m
    k_w: float     # shape factor (solid cylinder=1/2, etc.)

@dataclass
class ShotResult:
    v_p: float          # m/s
    omega_p: float      # rad/s
    omega_wi: float     # rad/s
    omega_wf: float     # rad/s
    ratio: float        # omega_wi/omega_wf
    eta: float          # mechanical efficiency (0..1)

def shooter_ratio(params: FlywheelShooterParams) -> float:
    """
    omega_wi/omega_wf = 1 + ((1 + k_p)/(4*k_w)) * (m_p/m_w)
    Source: FlywheelShooter notebook (shape-factor form).
    """
    return 1.0 + ((1.0 + params.k_p) / (4.0 * params.k_w)) * (params.m_p / params.m_w)

def solve_for_wheel_speeds_given_v(params: FlywheelShooterParams, v_p: float) -> ShotResult:
    """
    Uses:
      omega_wf = 2*v_p/r_w
      omega_wi = ratio * omega_wf
      omega_p = v_p/r_p
      eta = omega_wf/(omega_wf + omega_wi)
    """
    ratio = shooter_ratio(params)
    omega_wf = 2.0 * v_p / params.r_w
    omega_wi = ratio * omega_wf
    omega_p = v_p / params.r_p
    eta = omega_wf / (omega_wf + omega_wi)
    return ShotResult(v_p=v_p, omega_p=omega_p, omega_wi=omega_wi, omega_wf=omega_wf, ratio=ratio, eta=eta)

def solve_for_exit_v_given_omega_wi(params: FlywheelShooterParams, omega_wi: float) -> ShotResult:
    """
    Inverts:
      omega_wi = ratio * omega_wf = ratio * (2*v_p/r_w)
      => v_p = omega_wi * r_w / (2*ratio)
    """
    ratio = shooter_ratio(params)
    v_p = omega_wi * params.r_w / (2.0 * ratio)
    return solve_for_wheel_speeds_given_v(params, v_p)

# -------------------------
# Projectile motion model
# -------------------------
@dataclass
class ProjectileFlightParams:
    theta_deg: float     # launch angle (deg)
    h0: float            # initial height (m)
    g: float = 9.80665   # m/s^2

    # Optional quadratic drag
    use_drag: bool = False
    rho: float = 1.225      # air density kg/m^3
    Cd: float = 0.47        # sphere-ish default
    area: float = 0.01      # m^2  (set to pi*r^2 if you know radius)

def simulate_flight(v0: float, m: float, flight: ProjectileFlightParams,
                    dt: float = 0.002, t_max: float = 5.0):
    """
    2D flight simulation. If drag disabled -> analytic-like integration.
    If drag enabled -> numerical integration using Euler (simple & readable).
    Returns arrays t, x, y, vx, vy.
    """
    theta = np.deg2rad(flight.theta_deg)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    x, y = 0.0, flight.h0

    t_list, x_list, y_list, vx_list, vy_list = [], [], [], [], []
    t = 0.0

    while t <= t_max and y >= 0.0:
        t_list.append(t); x_list.append(x); y_list.append(y)
        vx_list.append(vx); vy_list.append(vy)

        ax, ay = 0.0, -flight.g

        if flight.use_drag:
            v = np.hypot(vx, vy)
            if v > 1e-9:
                Fd = 0.5 * flight.rho * flight.Cd * flight.area * v**2
                ax += -(Fd/m) * (vx / v)
                ay += -(Fd/m) * (vy / v)

        # Euler integration step
        vx += ax * dt
        vy += ay * dt
        x  += vx * dt
        y  += vy * dt
        t  += dt

    return (np.array(t_list), np.array(x_list), np.array(y_list),
            np.array(vx_list), np.array(vy_list))

# -------------------------
# Demo / Example usage
# -------------------------
if __name__ == "__main__":
    # Example parameters (EDIT THESE to your system)
    # k_w: solid cylinder = 0.5 ; k_p: solid sphere = 0.4 (2/5)
    params = FlywheelShooterParams(
        m_p=0.27,   # kg (example)
        r_p=0.06,   # m
        k_p=0.4,
        m_w=2.0,    # kg
        r_w=0.05,   # m
        k_w=0.5
    )

    # Choose either a desired exit speed OR an initial wheel speed
    desired_vp = 12.0  # m/s (example)
    shot = solve_for_wheel_speeds_given_v(params, desired_vp)

    print("=== Shot result (SI units) ===")
    print(f"Exit speed v_p      = {shot.v_p:.3f} m/s")
    print(f"Ball spin omega_p   = {shot.omega_p:.1f} rad/s")
    print(f"Wheel omega_wi      = {shot.omega_wi:.1f} rad/s ({radps_to_rpm(shot.omega_wi):.0f} rpm)")
    print(f"Wheel omega_wf      = {shot.omega_wf:.1f} rad/s ({radps_to_rpm(shot.omega_wf):.0f} rpm)")
    print(f"Speed ratio wi/wf   = {shot.ratio:.3f}")
    print(f"Efficiency eta      = {shot.eta*100:.1f} %")

    # Flight simulation (movement after leaving shooter)
    flight = ProjectileFlightParams(theta_deg=40.0, h0=1.0, use_drag=False)
    t, x, y, vx, vy = simulate_flight(shot.v_p, params.m_p, flight, dt=0.002, t_max=5.0)

    # Plot trajectory
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Projectile trajectory (after shooter exit)")
    plt.grid(True)

    # Plot wheel speed step (simple depiction)
    plt.figure()
    plt.step([0, 1], [shot.omega_wi, shot.omega_wf], where="post")
    plt.xticks([0, 1], ["Before shot", "After shot"])
    plt.ylabel("Wheel speed ω (rad/s)")
    plt.title("Wheel speed drop during shot (model)")
    plt.grid(True)

    plt.show()
