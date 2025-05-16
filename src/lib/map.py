import io
import folium


def map_to_html(m: folium.Map):
    buf = io.BytesIO()
    m.save(buf, close_file=False)
    return buf.getvalue().decode()