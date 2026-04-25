"""
RLTracker — Composant unifié de logging TensorBoard pour le projet Deep RL.

Usage basique:
    tracker = RLTracker("DQN", "GridWorld", config={"lr": 1e-3, "gamma": 0.99})
    for episode in range(num_episodes):
        # ... training loop ...
        tracker.log_episode(episode, score=score, loss=loss, epsilon=agent.epsilon, steps=steps)
    tracker.log_evaluation(num_episodes, avg_score=0.95, avg_steps=12.5)
    tracker.log_move_time(0.003)
    tracker.finish()

Lancer TensorBoard:
    tensorboard --logdir runs/
"""

import csv
import os
import time
from collections import deque
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


class RLTracker:
    """Composant unifié de logging pour tous les agents et environnements."""

    EVAL_MILESTONES = [1_000, 10_000, 100_000, 200_000, 300_000, 400_000, 500_000, 1_000_000]

    def __init__(
        self,
        agent_name: str,
        env_name: str,
        config: dict = None,
        run_name: str = None,
        log_dir: str = "runs",
        csv_dir: str = "results",
        smoothing_window: int = 100,
    ):
        self.agent_name = agent_name
        self.env_name = env_name
        self.config = config or {}
        self.smoothing_window = smoothing_window
        self.csv_dir = csv_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"{timestamp}"
        run_path = os.path.join(log_dir, env_name, agent_name, self.run_name)

        self.writer = SummaryWriter(log_dir=run_path)
        self._start_time = time.time()

        # Buffers pour moyennes glissantes
        self._score_buffer = deque(maxlen=smoothing_window)
        self._loss_buffer = deque(maxlen=smoothing_window)
        self._steps_buffer = deque(maxlen=smoothing_window)

        # Stockage pour export CSV
        self._evaluations = []
        self._move_time = None
        self._last_episode = 0

        # Log les hyperparamètres comme texte (visible dans TensorBoard)
        if self.config:
            config_text = "\n".join(f"- **{k}**: {v}" for k, v in self.config.items())
            self.writer.add_text("config", config_text, 0)

    # ------------------------------------------------------------------
    # Logging pendant l'entraînement
    # ------------------------------------------------------------------

    def log_episode(
        self,
        episode: int,
        score: float = None,
        loss: float = None,
        epsilon: float = None,
        steps: int = None,
        **extra_scalars,
    ):
        """Log les métriques d'un épisode d'entraînement.

        Args:
            episode: Numéro de l'épisode.
            score: Score obtenu (reward cumulé ou résultat de la partie).
            loss: Loss moyenne de l'épisode.
            epsilon: Taux d'exploration actuel.
            steps: Nombre de steps dans l'épisode (longueur de la partie).
            **extra_scalars: Métriques supplémentaires (ex: grad_norm=0.5).
        """
        self._last_episode = episode

        if score is not None:
            self.writer.add_scalar("train/score", score, episode)
            self._score_buffer.append(score)
            if len(self._score_buffer) >= self.smoothing_window:
                avg = sum(self._score_buffer) / len(self._score_buffer)
                self.writer.add_scalar("train/score_avg", avg, episode)

        if loss is not None:
            self.writer.add_scalar("train/loss", loss, episode)
            self._loss_buffer.append(loss)
            if len(self._loss_buffer) >= self.smoothing_window:
                avg = sum(self._loss_buffer) / len(self._loss_buffer)
                self.writer.add_scalar("train/loss_avg", avg, episode)

        if epsilon is not None:
            self.writer.add_scalar("train/epsilon", epsilon, episode)

        if steps is not None:
            self.writer.add_scalar("train/steps", steps, episode)
            self._steps_buffer.append(steps)
            if len(self._steps_buffer) >= self.smoothing_window:
                avg = sum(self._steps_buffer) / len(self._steps_buffer)
                self.writer.add_scalar("train/steps_avg", avg, episode)

        for key, value in extra_scalars.items():
            self.writer.add_scalar(f"train/{key}", value, episode)

    # ------------------------------------------------------------------
    # Logging d'évaluation (aux paliers du syllabus)
    # ------------------------------------------------------------------

    def log_evaluation(
        self,
        num_episodes_trained: int,
        avg_score: float,
        avg_steps: float = None,
        avg_reward: float = None,
        **extra_metrics,
    ):
        """Log les résultats d'évaluation à un palier d'entraînement.

        Args:
            num_episodes_trained: Nombre d'épisodes d'entraînement effectués.
            avg_score: Score moyen sur les parties d'évaluation.
            avg_steps: Longueur moyenne des parties (si durée variable).
            avg_reward: Reward moyen (si différent du score).
            **extra_metrics: Métriques supplémentaires.
        """
        eval_data = {
            "num_episodes_trained": num_episodes_trained,
            "avg_score": avg_score,
        }

        self.writer.add_scalar("eval/avg_score", avg_score, num_episodes_trained)

        if avg_steps is not None:
            self.writer.add_scalar("eval/avg_steps", avg_steps, num_episodes_trained)
            eval_data["avg_steps"] = avg_steps

        if avg_reward is not None:
            self.writer.add_scalar("eval/avg_reward", avg_reward, num_episodes_trained)
            eval_data["avg_reward"] = avg_reward

        for key, value in extra_metrics.items():
            self.writer.add_scalar(f"eval/{key}", value, num_episodes_trained)
            eval_data[key] = value

        self._evaluations.append(eval_data)

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    def log_move_time(self, avg_time_per_move: float):
        """Log le temps moyen pour exécuter un coup (en secondes)."""
        self._move_time = avg_time_per_move
        self.writer.add_scalar("perf/move_time_ms", avg_time_per_move * 1000, 0)

    # ------------------------------------------------------------------
    # HParams (pour le dashboard HParams de TensorBoard)
    # ------------------------------------------------------------------

    def _log_hparams(self):
        """Log les hyperparamètres avec les métriques finales pour le HParams dashboard."""
        if not self.config or not self._evaluations:
            return

        # Dernière évaluation comme métrique finale
        last_eval = self._evaluations[-1]
        metrics = {"hparam/avg_score": last_eval["avg_score"]}
        if "avg_steps" in last_eval:
            metrics["hparam/avg_steps"] = last_eval["avg_steps"]
        if self._move_time is not None:
            metrics["hparam/move_time"] = self._move_time

        # Filtrer les valeurs non-scalaires du config
        hparams = {}
        for k, v in self.config.items():
            if isinstance(v, (int, float, str, bool)):
                hparams[k] = v

        self.writer.add_hparams(hparams, metrics, run_name=".")

    # ------------------------------------------------------------------
    # Export CSV (compatibilité avec results_plot.py)
    # ------------------------------------------------------------------

    def _export_csv(self):
        """Exporte les résultats en CSV compatible avec le système existant."""
        if not self._evaluations:
            return

        os.makedirs(self.csv_dir, exist_ok=True)
        filename = f"{self.env_name.lower()}_{self.agent_name.lower()}_results.csv"
        filepath = os.path.join(self.csv_dir, filename)

        elapsed = time.time() - self._start_time

        rows = []
        for eval_data in self._evaluations:
            row = {
                "config_name": self.run_name,
                "agent": self.agent_name,
                "env": self.env_name,
                "num_episodes": eval_data["num_episodes_trained"],
                "avg_score": eval_data["avg_score"],
                "execution_time_sec": round(elapsed, 2),
            }
            if "avg_steps" in eval_data:
                row["avg_steps"] = eval_data["avg_steps"]
            if self._move_time is not None:
                row["avg_move_time_sec"] = self._move_time
            # Ajouter les hyperparamètres
            for k, v in self.config.items():
                if isinstance(v, (int, float, str, bool)) and k not in row:
                    row[k] = v
            rows.append(row)

        # Append si le fichier existe déjà (pour accumuler les runs)
        file_exists = os.path.exists(filepath)
        existing_fields = set()
        if file_exists:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_fields = set(reader.fieldnames or [])

        all_fields = list(rows[0].keys())
        if file_exists:
            # Garder l'ordre existant, ajouter les nouvelles colonnes à la fin
            new_fields = [f for f in all_fields if f not in existing_fields]
            all_fields = list(existing_fields) + new_fields

            # Relire les données existantes
            existing_rows = []
            with open(filepath, "r", encoding="utf-8") as f:
                existing_rows = list(csv.DictReader(f))
            rows = existing_rows + rows

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finish(self):
        """Finalise le tracking : log les hparams, exporte le CSV, ferme le writer."""
        elapsed = time.time() - self._start_time
        self.writer.add_scalar("perf/total_time_sec", elapsed, 0)

        self._log_hparams()
        self._export_csv()
        self.writer.flush()
        self.writer.close()

        print(f"[RLTracker] {self.env_name}/{self.agent_name}/{self.run_name} — "
              f"terminé en {elapsed:.1f}s")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False