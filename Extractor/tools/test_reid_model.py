import cv2
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from torchreid.reid.utils import FeatureExtractor
from glob import glob
import random
import os
import io
import base64
from PIL import Image
from dash import Dash, dcc, html, Input, Output, no_update, callback


def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

@callback(
    Output("graph-tooltip-5", "show"),
    Output("graph-tooltip-5", "bbox"),
    Output("graph-tooltip-5", "children"),
    Input("graph-5", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]

    im_matrix = np_list[num]
    im_url = np_image_to_base64(im_matrix)
    children = [
        html.Div([
            html.Img(
                src=im_url,
                style={"width": "50px", 'display': 'block', 'margin': '0 auto'},
            ),
            html.P(str(os.path.basename(image_paths_1016[num])), style={'font-weight': 'bold'})
        ])
    ]

    return True, bbox, children


image_paths = glob(os.path.join(r"C:\AICUP\ReId_dataset_2", "*", "*.jpg"))
for cam in [0, 1, 2, 3, 5, 6, 7]:
    image_paths_1016 = []
    np_list = []

    # 1016 images
    for image_name in image_paths:
        image_name_ = image_name.split("\\")[-1]
        date, time_start, time_finish, camera_id, _ = image_name_.split("_")
        if date == "1016" and time_start == "190000" and camera_id == f"{cam}":  # and time_start == "150000" and camera_id == "0"
            image_paths_1016.append(image_name)
            mat_image = cv2.imread(image_name)
            np_image = np.array(mat_image)
            np_list.append(np_image)

    tracker_ids = [path.split("\\")[-2] for path in image_paths_1016]
    unique_tracker_ids = list(set(tracker_ids))
    color_map = {tracker_id: f'#{random.randint(0, 0xFFFFFF):06x}' for tracker_id in unique_tracker_ids}

    print("=> Total images: ", len(image_paths_1016))
    # Feature extraction model
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=r'D:\Jonathan\AI_project\ObjectTracking\code\yolov9\ReId\log\osnet_x1_0\model\model.pth.tar-50',
        device='cuda'
    )
    print("=> Get features")
    feature_results = extractor(image_paths_1016)
    flattened_features = feature_results.cpu().numpy()  # GPU
    # flattened_features = np.concatenate(feature_results, axis=0)  # CPU

    print("=> t-SNE")
    # reduce dimension
    tsne = TSNE(n_components=3, random_state=0)
    reduced_features = tsne.fit_transform(flattened_features)
    # np.save("1016_morning_4_reduce_features.npy", reduced_features)

    print("=> Visualization")
    fig = go.Figure()
    for tracker_id in unique_tracker_ids:
        filtered_features = []
        filtered_name = []
        for i, (img_path, feature) in enumerate(zip(image_paths_1016, reduced_features)):
            if tracker_ids[i] == tracker_id:
                filtered_features.append(feature)
                filtered_name.append(os.path.basename(img_path))

        # filtered_features = [feature for i, feature in enumerate(reduced_features) if tracker_ids[i] == tracker_id]

        fig.add_trace(go.Scatter3d(
            x=[f[0] for f in filtered_features],
            y=[f[1] for f in filtered_features],
            z=[f[2] for f in filtered_features],
            mode='markers+text',
            marker=dict(color=color_map[tracker_id]),
            text=[tracker_id] * len(filtered_features),
            textposition='top center',
            name=tracker_id,
            hovertemplate='"%{customdata}"',
            customdata=[f for f in filtered_name]
        ))
    fig.write_html(f"1016_night_{cam}.html")
# fig.show()


# fig.update_traces(
#     hoverinfo="none",
#     hovertemplate=None,
# )
#
# app = Dash(__name__)
# app.layout = html.Div(
#     className="container",
#     children=[
#         dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
#         dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
#     ],
# )
#
# if __name__ == '__main__':
#     app.run(debug=True)
