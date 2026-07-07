from envs import make_env


def run_random_agent(episodes=50, render=False):
    # render=False时  不显示窗口，运行会更快。
    render_mode = "human" if render else None
    env = make_env(render_mode=render_mode)
    rewards = []
    steps_list = []

    for episode in range(episodes):
        # 每个 episode 都要先重置环境。
        state, info = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            # 随机智能体：不看 state，直接随机选动作。
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated
            state = next_state
            step += 1

        rewards.append(total_reward)
        steps_list.append(step)
        print(f"Episode {episode + 1}: steps={step}, reward={total_reward:.2f}")

    env.close()

    # 随机基线：训练 DQN 时，用它作为最低水平对照。
    avg_reward = sum(rewards) / len(rewards)
    avg_steps = sum(steps_list) / len(steps_list)

    print("\n随机智能体统计结果")
    print(f"Episodes: {episodes}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Best reward: {max(rewards):.2f}")
    print(f"Worst reward: {min(rewards):.2f}")
    print(f"Average steps: {avg_steps:.1f}")


if __name__ == "__main__":
    run_random_agent(episodes=50, render=False)
