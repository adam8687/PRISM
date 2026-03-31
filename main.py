import robosuite as suite

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20
)

obs = env.reset()

print("Observation keys:", obs.keys())

for step in range(200):
    action = [0] * env.action_dim
    obs, reward, done, info = env.step(action)
    
    print(f"Step {step}, reward={reward}")
    
    env.render()

    if done:
        break

env.close()
