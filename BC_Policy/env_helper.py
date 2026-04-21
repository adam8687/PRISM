"""
Helper to get proper controller configuration for RoboCasa/Robosuite environments.
"""

import robosuite as suite
from robosuite.controllers import load_controller_config


def get_controller_config(controller_type="OSC_POSE"):
    """
    Load default controller configuration from robosuite.
    
    Args:
        controller_type: Type of controller (e.g., "OSC_POSE", "OSC_POSITION", "IK_POSE")
    
    Returns:
        Complete controller configuration dictionary
    """
    # Load the default config from robosuite
    controller_config = load_controller_config(default_controller=controller_type)
    
    return controller_config


def create_pickegg_env(
    env_class,
    use_damage=False,
    controller_type="OSC_POSE",
    robots="PandaMobile",
    has_renderer=False,
    has_offscreen_renderer=False,
    camera_names="robot0_agentview_center",
    horizon=500,
    reward_shaping=True,
):
    """
    Create PickEgg environment with proper controller configuration.
    
    Args:
        env_class: The environment class (PickEgg or DamageablePickEgg)
        use_damage: Whether to use damageable variant
        controller_type: Type of controller to use
        robots: Robot type
        has_renderer: Enable on-screen rendering
        has_offscreen_renderer: Enable off-screen rendering
        camera_names: Camera name(s)
        horizon: Episode horizon
        reward_shaping: Enable reward shaping
    
    Returns:
        Initialized environment
    """
    # Get default controller config
    controller_config = get_controller_config(controller_type)
    
    # Create environment config
    env_config = {
        "robots": robots,
        "controller_configs": controller_config,
        "has_renderer": has_renderer,
        "has_offscreen_renderer": has_offscreen_renderer,
        "camera_names": camera_names,
        "reward_shaping": reward_shaping,
        "horizon": horizon,
    }
    
    # Create and return environment
    env = env_class(**env_config)
    
    return env


if __name__ == "__main__":
    # Test loading configs
    print("Available controller types and their configs:\n")
    
    for controller_type in ["OSC_POSE", "OSC_POSITION", "IK_POSE", "JOINT_POSITION"]:
        try:
            config = get_controller_config(controller_type)
            print(f"{controller_type}:")
            print(f"  Parameters: {list(config.keys())}")
            print()
        except Exception as e:
            print(f"{controller_type}: Error - {e}\n")