from envs import make_env


def main():
    # render_mode="human" 会打开可视化窗口。
    env = make_env(render_mode="human")

    print("状态空间:", env.observation_space)
    print("动作空间:", env.action_space)

    for episode in range(3):
        # reset() 开始新一局，返回初始状态。
        state, info = env.reset()
        total_reward = 0

        done = False
        step = 0

        while not done:
            # 随机动作只用于测试环境，不是训练后的策略。
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated
            state = next_state
            step += 1

        print(f"Episode {episode + 1}: steps={step}, reward={total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
