from deep_sort_realtime.deepsort_tracker import DeepSort
import os

class Tracker:
    def __init__(self, max_age=15, n_init=3, max_cosine_distance=0.4, nn_budget=50):
        """Initialize the DeepSort tracker with specified parameters."""
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget
        )

    def update(self, detections, frame):
        """Update the tracker with new detections."""
        return self.tracker.update_tracks(detections, frame=frame)