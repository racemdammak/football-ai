import supervision as sv
from tqdm import tqdm
import torch
from team_assignment import TeamClassifier

def start_detection(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH, PLAYER_DETECTION_MODEL):
    
    BALL_ID = 0
    GOALKEEPER_ID = 1
    PLAYER_ID = 2
    REFEREE_ID = 3

    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=25,
        height=21,
        outline_thickness=1
    )

    tracker = sv.ByteTrack()
    tracker.reset()

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info)
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu" 

    print(f"Using device: {device}")

    team_classifier = TeamClassifier(player_detection_model=PLAYER_DETECTION_MODEL, device=device)
    crops = team_classifier.crop(SOURCE_VIDEO_PATH)
    team_classifier.fit(crops)

    with video_sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            result = PLAYER_DETECTION_MODEL(frame, conf=0.3)[0]
            detections = sv.Detections.from_ultralytics(result)

            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            all_detections = detections[detections.class_id != BALL_ID]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
            all_detections = tracker.update_with_detections(detections=all_detections)

            goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
            players_detections = all_detections[all_detections.class_id == PLAYER_ID]
            referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            players_detections.class_id = team_classifier.predict(players_crops)

            goalkeepers_detections.class_id = team_classifier.resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)

            referees_detections.class_id -= 1

            all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])

            labels = [f"#{tracker_id}"for tracker_id in all_detections.tracker_id]

            all_detections.class_id = all_detections.class_id.astype(int)

            annotated_frame = frame.copy()
            annotated_frame = ellipse_annotator.annotate(scene=annotated_frame,detections=all_detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame,detections=all_detections,labels=labels)
            annotated_frame = triangle_annotator.annotate(scene=annotated_frame,detections=ball_detections)

            video_sink.write_frame(annotated_frame)
