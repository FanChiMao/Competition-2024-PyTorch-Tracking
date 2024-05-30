
class MOT_TXT:
    def __init__(self, frame_id, track_id, x1, y1, x2, y2, object_class):
        self.frame_id = frame_id
        self.track_id = track_id
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.object_class = object_class

def write_txt_by_line(txt_file_path, mot_txt: MOT_TXT, x1y1wh=True):
    with open(txt_file_path, "a+") as f:
        if x1y1wh:
            w = mot_txt.x2 - mot_txt.x1
            h = mot_txt.y2 - mot_txt.y1
            xc = mot_txt.x1 + w / 2
            yc = mot_txt.y1 + h / 2
            f.write(f"{mot_txt.frame_id},{mot_txt.track_id},{mot_txt.x1},{mot_txt.y1},{w},{h},"
                    f"{mot_txt.object_class},-1,-1,-1\n")
        else:  # x1y1x2y2
            f.write(f"{mot_txt.frame_id},{mot_txt.track_id},{mot_txt.x1},{mot_txt.y1},{mot_txt.x2},{mot_txt.y2},"
                    f"{mot_txt.object_class},-1,-1,-1\n")
