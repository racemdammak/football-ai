import supervision as sv
from tqdm import tqdm
import torch
import numpy as np
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
    sv.plot_images_grid(crops[:100], grid_size=(10, 10))
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

            # Traitement des joueurs
            if len(players_detections) > 0:
                players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
                team_predictions = team_classifier.predict(players_crops)
                
                # Debug: vérifier les dimensions
                print(f"Players détections: {len(players_detections)}")
                print(f"Team predictions shape: {team_predictions.shape if hasattr(team_predictions, 'shape') else len(team_predictions)}")
                
                # S'assurer que les dimensions correspondent
                if len(team_predictions) == len(players_detections):
                    players_detections.class_id = team_predictions.astype(np.int32)
                else:
                    print(f"Dimension mismatch! Players: {len(players_detections)}, Predictions: {len(team_predictions)}")
                    # Fallback: assigner classe 0 à tous les joueurs
                    players_detections.class_id = np.zeros(len(players_detections), dtype=np.int32)

            # Traitement des gardiens de but
            if len(goalkeepers_detections) > 0:
                if len(players_detections) > 0:
                    goalkeeper_team_ids = team_classifier.resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)
                    # S'assurer que les dimensions correspondent
                    if len(goalkeeper_team_ids) == len(goalkeepers_detections):
                        goalkeepers_detections.class_id = goalkeeper_team_ids.astype(np.int32)
                    else:
                        # Fallback: assigner classe 0 aux gardiens
                        goalkeepers_detections.class_id = np.zeros(len(goalkeepers_detections), dtype=np.int32)
                else:
                    # Pas de joueurs pour déterminer l'équipe, assigner classe 0
                    goalkeepers_detections.class_id = np.zeros(len(goalkeepers_detections), dtype=np.int32)

            # Traitement des arbitres
            if len(referees_detections) > 0:
                # Assigner la classe 2 aux arbitres
                referees_detections.class_id = np.full(len(referees_detections), 2, dtype=np.int32)

            # Debug: vérifier les class_id avant fusion
            detections_to_merge = []
            if len(players_detections) > 0:
                print(f"Players class_id shape: {players_detections.class_id.shape}, len: {len(players_detections)}")
                detections_to_merge.append(players_detections)
            if len(goalkeepers_detections) > 0:
                print(f"Goalkeepers class_id shape: {goalkeepers_detections.class_id.shape}, len: {len(goalkeepers_detections)}")
                detections_to_merge.append(goalkeepers_detections)
            if len(referees_detections) > 0:
                print(f"Referees class_id shape: {referees_detections.class_id.shape}, len: {len(referees_detections)}")
                detections_to_merge.append(referees_detections)

            # Fusionner seulement s'il y a des détections
            if detections_to_merge:
                try:
                    all_detections = sv.Detections.merge(detections_to_merge)
                except Exception as e:
                    print(f"Erreur lors de la fusion: {e}")
                    # En cas d'erreur, créer des détections vides
                    all_detections = sv.Detections.empty()
                labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]
                
                # S'assurer que class_id est de type int
                all_detections.class_id = all_detections.class_id.astype(int)
            else:
                # Si aucune détection, créer un objet vide
                all_detections = sv.Detections.empty()
                labels = []

            annotated_frame = frame.copy()
            
            # Annoter seulement s'il y a des détections
            if len(all_detections) > 0:
                annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=all_detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=all_detections, labels=labels)
            
            # Annoter le ballon séparément
            if len(ball_detections) > 0:
                annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)

            video_sink.write_frame(annotated_frame)