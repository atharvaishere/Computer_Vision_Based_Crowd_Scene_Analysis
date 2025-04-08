import cv2
import os

class VideoProcessor:
    def __init__(self, video_path, output_folder, detector, tracker, analyzer, desired_fps=10):
        """Initialize the VideoProcessor with video parameters and components."""
        self.video_path = video_path
        self.output_folder = output_folder
        self.detector = detector
        self.tracker = tracker
        self.analyzer = analyzer
        self.desired_fps = desired_fps
        self.cap = cv2.VideoCapture(video_path)
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.original_fps
        if self.duration < 1:
            self.original_fps = 125
            self.duration = self.total_frames / self.original_fps
        self.target_frame_count = int(self.desired_fps * self.duration)
        self.frame_skip = max(1, int(round(self.original_fps / self.desired_fps)))
        self.out = cv2.VideoWriter(
            os.path.join(output_folder, "output-behavior-analysis.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.desired_fps,
            (512, 384)
        )
        self.frame_counter = 0
        self.processed_frames = 0
        self.interaction_colors = {
            "Handshake": (0, 255, 0),
            "Pushing": (0, 0, 255),
            "Wrestling": (255, 0, 0),
            "Close Proximity": (255, 255, 0),
            "Interaction": (255, 0, 255)
        }

    def process(self):
        """Process the video frame by frame."""
        while self.cap.isOpened() and self.processed_frames < self.target_frame_count:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.frame_counter % self.frame_skip != 0:
                self.frame_counter += 1
                continue
            frame = cv2.resize(frame, (512, 384))
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections, frame)
            self.analyzer.update_tracks(tracks, self.processed_frames)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                track_id = track.track_id
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                info_text = f"ID: {track_id} ({x2-x1}x{y2-y1})"
                cv2.putText(frame, info_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            for id1, id2, interaction_type, confidence in self.analyzer.get_interactions():
                center1 = self.analyzer._box_center(self.analyzer.track_history[id1]['boxes'][-1])
                center2 = self.analyzer._box_center(self.analyzer.track_history[id2]['boxes'][-1])
                color = self.interaction_colors.get(interaction_type, (255, 255, 255))
                cv2.line(frame, center1, center2, color, 2)
                mid_point = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
                cv2.putText(frame, f"{interaction_type} {confidence*100:.0f}%", mid_point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            self.out.write(frame)
            self.processed_frames += 1
            print(f"Processed frame {self.processed_frames}/{self.target_frame_count}")
            self.frame_counter += 1
        self.cap.release()
        self.out.release()

    def generate_reports(self):
        """Generate and save behavior reports and interaction plots."""
        behavior_report = self.analyzer.get_behavior_analysis()
        report_path = os.path.join(self.output_folder, "behavior_report.txt")
        with open(report_path, 'w') as f:
            f.write("Enhanced Behavior Analysis Report\n")
            f.write(f"Video: {self.video_path}\n")
            f.write(f"Total Frames Processed: {self.processed_frames}\n\n")
            for track_id, analysis in behavior_report.items():
                f.write(f"\nTrack {track_id} Analysis:\n")
                f.write(f"- Movement Pattern: {analysis['movement_pattern']}\n")
                f.write(f"- Average Speed: {analysis['avg_speed']:.2f} pixels/frame\n")
                if analysis['interactions']:
                    f.write("- Significant Interactions:\n")
                    for partner, details in analysis['interactions'].items():
                        f.write(f"  - With ID {partner}: {details['type']} "
                                f"(Duration: {details['duration']} frames, "
                                f"Avg Distance: {details['avg_distance']:.1f}px)\n")
                else:
                    f.write("- No significant interactions detected\n")
        self.analyzer.generate_interaction_plots(self.output_folder)