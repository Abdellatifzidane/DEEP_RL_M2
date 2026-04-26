import os
import time
import torch

from agents.policy_gradient.reinforce_mean_baseline import ReinforceMeanBaselineAgent
from environnements.quarto.quatro import Quatro
from evaluate.tracker import RLTracker


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "agent_name": "ReinforceMeanBaseline",
    "env_name": "Quatro",

    "gamma": 0.99,
    "learning_rate": 1e-3,

    # RUN FINAL
    "max_training_episodes": 300_000,

    # évaluation plus stable
    "num_eval_episodes": 200,

    # checkpoints réguliers
    "checkpoint_every": 10_000,

    "checkpoint_dir": "checkpoints",
    "log_dir": "runs",
    "csv_dir": "results",
}


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(agent, episode, config):
    checkpoint_dir = os.path.join(
        config["checkpoint_dir"],
        config["env_name"].lower(),
        config["agent_name"].lower()
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "episode": episode,
        "config": config,
        "policy_state_dict": agent.policy_network.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
    }

    filepath = os.path.join(checkpoint_dir, f"episode_{episode}.pt")
    torch.save(checkpoint, filepath)

    print(f"[CHECKPOINT] Modèle sauvegardé : {filepath}")


# ============================================================
# EVALUATION
# ============================================================

def evaluate_agent(agent, env_class, num_eval_episodes):
    total_score = 0.0
    total_steps = 0
    total_moves = 0
    total_move_time = 0.0

    for _ in range(num_eval_episodes):
        env = env_class()
        env.reset()

        episode_steps = 0

        while not env.is_terminal():
            start_time = time.time()

            action = agent.agir(env)

            end_time = time.time()
            total_move_time += end_time - start_time
            total_moves += 1

            if action is None:
                break

            env.step(action)
            episode_steps += 1

        total_score += env.get_score()
        total_steps += episode_steps

    avg_score = total_score / num_eval_episodes
    avg_steps = total_steps / num_eval_episodes
    avg_move_time = total_move_time / total_moves if total_moves > 0 else 0.0

    return avg_score, avg_steps, avg_move_time


# ============================================================
# TRAINING
# ============================================================

def main():
    env = Quatro()

    taille_etat = len(env.get_state())
    nombre_actions = env.action_size

    agent = ReinforceMeanBaselineAgent(
        taille_etat=taille_etat,
        nombre_actions=nombre_actions,
        gamma=CONFIG["gamma"],
        learning_rate=CONFIG["learning_rate"],
    )

    eval_milestones = [
        milestone
        for milestone in RLTracker.EVAL_MILESTONES
        if milestone <= CONFIG["max_training_episodes"]
    ]

    tracker = RLTracker(
        agent_name=CONFIG["agent_name"],
        env_name=CONFIG["env_name"],
        config=CONFIG,
        log_dir=CONFIG["log_dir"],
        csv_dir=CONFIG["csv_dir"],
    )

    saved_checkpoints = set()

    try:
        for episode in range(1, CONFIG["max_training_episodes"] + 1):
            result = agent.jouer_un_episode_et_apprendre(env)

            tracker.log_episode(
                episode=episode,
                score=result["reward_total_episode"],
                loss=result["loss"],
                steps=result["nombre_etapes_episode"],
            )

            # =========================
            # EVALUATION
            # =========================
            if episode in eval_milestones:
                avg_score, avg_steps, avg_move_time = evaluate_agent(
                    agent=agent,
                    env_class=Quatro,
                    num_eval_episodes=CONFIG["num_eval_episodes"],
                )

                tracker.log_evaluation(
                    num_episodes_trained=episode,
                    avg_score=avg_score,
                    avg_steps=avg_steps,
                )

                tracker.log_move_time(avg_move_time)

                print(
                    f"[EVAL] Episode {episode} | "
                    f"avg_score={avg_score:.4f} | "
                    f"avg_steps={avg_steps:.2f} | "
                    f"avg_move_time={avg_move_time:.6f}s"
                )

                save_checkpoint(agent, episode, CONFIG)
                saved_checkpoints.add(episode)

            # =========================
            # CHECKPOINT REGULIER
            # =========================
            if (
                episode % CONFIG["checkpoint_every"] == 0
                and episode not in saved_checkpoints
            ):
                save_checkpoint(agent, episode, CONFIG)
                saved_checkpoints.add(episode)

    finally:
        tracker.finish()


if __name__ == "__main__":
    main()