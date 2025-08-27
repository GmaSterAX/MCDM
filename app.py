import io
import base64
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import numpy as np
from Algorithms.Entropy import entropy
from Algorithms.SAW import SAW
from Algorithms.AHP import AHP
from Algorithms.TOPSIS import TOPSIS
from Algorithms.VIKOR import VIKOR

app = Flask(__name__)


def plot_bar(data, title="Bar Chart", xlabel="Alternatives", ylabel="Value"):
    fig, ax = plt.subplots()
    ax.bar(data.keys(), data.values(), color='skyblue')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def update_dict_keys(original_dict, new_keys):
    """Dictionary'nin key'lerini yeni isimlerle günceller"""
    if not isinstance(original_dict, dict):
        return original_dict

    updated_dict = {}
    for i, (old_key, value) in enumerate(original_dict.items()):
        if i < len(new_keys):
            updated_dict[new_keys[i]] = value
        else:
            updated_dict[old_key] = value
    return updated_dict


def safe_update_ranking(ranking_list, alt_names):
    """Ranking listesini güvenli şekilde günceller"""
    if not ranking_list:
        return ranking_list

    updated_ranking = []
    for rank in ranking_list:
        if isinstance(rank, str) and rank.startswith('A') and len(rank) > 1:
            try:
                index = int(rank[1:]) - 1
                if 0 <= index < len(alt_names):
                    updated_ranking.append(alt_names[index])
                else:
                    updated_ranking.append(rank)
            except (ValueError, IndexError):
                updated_ranking.append(rank)
        else:
            updated_ranking.append(str(rank))  # Güvenli string dönüşümü
    return updated_ranking


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    # Form verilerini al
    alt_names_raw = request.form.getlist("alternatives")
    crit_names_raw = request.form.getlist("criteria")
    crit_types = request.form.getlist("criteria_type")
    method = request.form.get("method")

    alt_count = len(alt_names_raw)
    crit_count = len(crit_names_raw)

    # Boş alternatif isimlerini otomatik isimlendirme
    alt_names = []
    for i, name in enumerate(alt_names_raw):
        if name.strip():  # Eğer isim boş değilse
            alt_names.append(name.strip())
        else:  # Eğer boşsa otomatik isimlendirme
            alt_names.append(f"A{i + 1}")

    # Boş kriter isimlerini otomatik isimlendirme
    crit_names = []
    for i, name in enumerate(crit_names_raw):
        if name.strip():  # Eğer isim boş değilse
            crit_names.append(name.strip())
        else:  # Eğer boşsa otomatik isimlendirme
            crit_names.append(f"C{i + 1}")

    # Karar matrisi topla
    decision_matrix = np.zeros((alt_count, crit_count))
    for i in range(alt_count):
        for j in range(crit_count):
            decision_matrix[i, j] = float(request.form.get(f"matrix_{i}_{j}"))

    # Hesaplama
    if method == "Entropy":
        w = entropy(decision_matrix)
        results = {"weights": {crit_names[i]: w[i] for i in range(len(w))}}
    elif method == "SAW":
        saw = SAW(decision_matrix, crit_types)
        results = saw.get_results()
        # SAW sonuçlarında alternatif isimlerini güncelle
        if "scores" in results:
            results["scores"] = update_dict_keys(results["scores"], alt_names)
        if "ranking" in results:
            results["ranking"] = safe_update_ranking(results["ranking"], alt_names)
    elif method == "AHP":
        n = len(crit_types)
        pairwise_matrix = np.ones((n, n))
        ahp = AHP(pairwise_matrix, decision_matrix, crit_types)
        results = ahp.get_results()
        # AHP sonuçlarında isim güncellemelerini yap
        if "weights" in results:
            results["weights"] = update_dict_keys(results["weights"], crit_names)
        if "scores" in results:
            results["scores"] = update_dict_keys(results["scores"], alt_names)
        if "ranking" in results:
            results["ranking"] = safe_update_ranking(results["ranking"], alt_names)
    elif method == "TOPSIS":
        topsis = TOPSIS(decision_matrix, crit_types)
        results = topsis.get_results()
        # TOPSIS sonuçlarında isim güncellemelerini yap
        if "closeness_coefficients" in results:
            results["closeness_coefficients"] = update_dict_keys(results["closeness_coefficients"], alt_names)
        if "ranking" in results:
            results["ranking"] = safe_update_ranking(results["ranking"], alt_names)
    elif method == "VIKOR":
        vikor = VIKOR(decision_matrix, crit_types)
        results = vikor.get_results()
        # VIKOR sonuçlarında isim güncellemelerini yap
        if "q_values" in results:
            results["q_values"] = update_dict_keys(results["q_values"], alt_names)
        if "ranking" in results:
            results["ranking"] = safe_update_ranking(results["ranking"], alt_names)
    else:
        results = {}

    # Grafikler
    graph_weights = None
    if "weights" in results:
        graph_weights = plot_bar(results["weights"], title="Kriter Ağırlıkları", ylabel="Ağırlık")

    graph_scores = None
    if "scores" in results:
        graph_scores = plot_bar(results["scores"], title="Alternatif Skorları", ylabel="Skor")
    elif "closeness_coefficients" in results:
        graph_scores = plot_bar(results["closeness_coefficients"], title="Yakınlık Katsayıları", ylabel="Değer")
    elif "q_values" in results:
        graph_scores = plot_bar(results["q_values"], title="Q Değerleri (VIKOR)", ylabel="Q")

    return render_template("result.html",
                           method=method,
                           results=results,
                           graph_weights=graph_weights,
                           graph_scores=graph_scores,
                           criteria=crit_names,
                           alternatives=alt_names)


if __name__ == "__main__":
    app.run(debug=True)