from my_agent_3 import RayAgent
import soccer_twos

env = soccer_twos.make()
agent = RayAgent(env)

obs = env.reset()
dones = {"__all__": False}
step = 0

while not dones["__all__"] and step < 200:
    actions = agent.act(obs)
    obs, rewards, dones, info = env.step(actions)
    print(f"step={step}, actions={actions}, rewards={rewards}, dones={dones}")
    step += 1