"""
╔══════════════════════════════════════════════════════════╗
║   VGGFace — Reconhecimento Facial em Tempo Real          ║
║   Dependências: torch torchvision opencv-python          ║
║                 Pillow numpy scikit-learn h5py           ║
║                                                          ║
║   Uso:                                                   ║
║     # Cadastrar faces antes de rodar:                    ║
║     python realtime_recognition.py --register            ║
║                                                          ║
║     # Rodar reconhecimento em tempo real:                ║
║     python realtime_recognition.py                       ║
║                                                          ║
║   Atalhos na janela OpenCV:                              ║
║     Q       → sair                                       ║
║     S       → salvar screenshot                          ║
║     R       → cadastrar face do frame atual              ║
║     ESPAÇO  → pausar / retomar                           ║
╚══════════════════════════════════════════════════════════╝
"""
#PARA REGISTRAR - python 3-VGGFace_Pytorch_WebCam.py --register
#PARA TESTAR - python 3-VGGFace_Pytorch_WebCam.py --h5 vgg_face_weights_Pytorch.h5 --threshold 0.65 --camera 1

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import h5py
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIGURAÇÕES — edite conforme necessário
# ──────────────────────────────────────────────
H5_PATH      = "vggface.h5"          # caminho para os pesos
DB_PATH      = "face_database.npz"   # banco de identidades (criado automaticamente)
CAMERA_INDEX = 0                     # índice da câmera (0 = padrão)
FRAME_SKIP   = 3                     # processa embedding a cada N frames
THRESHOLD    = 0.60                  # limiar de similaridade de cosseno
NUM_CLASSES  = 2622
# ──────────────────────────────────────────────


# ══════════════════════════════════════════════
# 1. MODELO
# ══════════════════════════════════════════════

class VGGFace(nn.Module):
    def __init__(self, num_classes=2622):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   64,  3, padding=1), nn.ReLU(True),   # conv1_1
            nn.Conv2d(64,  64,  3, padding=1), nn.ReLU(True),   # conv1_2
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64,  128, 3, padding=1), nn.ReLU(True),   # conv2_1
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),   # conv2_2
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),   # conv3_1
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),   # conv3_2
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),   # conv3_3
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True),   # conv4_1
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),   # conv4_2
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),   # conv4_3
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),   # conv5_1
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),   # conv5_2
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),   # conv5_3
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(0.5),  # fc6
            nn.Linear(4096, 4096),         nn.ReLU(True), nn.Dropout(0.5),  # fc7
            nn.Linear(4096, num_classes),                                    # fc8
        )

    def forward(self, x, embedding=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if embedding:
            # retorna saída da fc7 (4096-d) como embedding
            for layer in self.classifier[:5]:
                x = layer(x)
            return x
        return self.classifier(x)


def load_model(h5_path, device):
    print(f"[1/2] Construindo modelo VGGFace...")
    model = VGGFace(NUM_CLASSES)

    layer_map = [
        ("conv1_1", model.features[0]),
        ("conv1_2", model.features[2]),
        ("conv2_1", model.features[5]),
        ("conv2_2", model.features[7]),
        ("conv3_1", model.features[10]),
        ("conv3_2", model.features[12]),
        ("conv3_3", model.features[14]),
        ("conv4_1", model.features[17]),
        ("conv4_2", model.features[19]),
        ("conv4_3", model.features[21]),
        ("conv5_1", model.features[24]),
        ("conv5_2", model.features[26]),
        ("conv5_3", model.features[28]),
        ("fc6",     model.classifier[0]),
        ("fc7",     model.classifier[3]),
        ("fc8",     model.classifier[6]),
    ]

    print(f"[2/2] Carregando pesos de '{h5_path}'...")
    with h5py.File(h5_path, "r") as f:
        ok = 0
        for name, layer in layer_map:
            wk, bk = f"{name}.weight", f"{name}.bias"
            if wk not in f:
                print(f"  ⚠  '{wk}' não encontrado — pulando.")
                continue
            layer.weight.data.copy_(torch.FloatTensor(np.array(f[wk])))
            if bk in f:
                layer.bias.data.copy_(torch.FloatTensor(np.array(f[bk])))
            ok += 1

    print(f"   {ok}/{len(layer_map)} camadas carregadas.\n")
    model.to(device).eval()
    return model


# ══════════════════════════════════════════════
# 2. PRÉ-PROCESSAMENTO
# ══════════════════════════════════════════════

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.367, 0.411, 0.507], std=[1.0, 1.0, 1.0]),
])


def get_embedding(face_bgr, model, device):
    """Recebe crop de face em BGR (numpy), retorna embedding L2-normalizado."""
    pil = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    t   = _transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(t, embedding=True).cpu().numpy().squeeze()
    return emb / (np.linalg.norm(emb) + 1e-10)


# ══════════════════════════════════════════════
# 3. BANCO DE IDENTIDADES
# ══════════════════════════════════════════════

class FaceDatabase:
    def __init__(self, threshold=THRESHOLD):
        self.db        = {}           # {nome: [emb, ...]}
        self.threshold = threshold

    def register(self, name, emb):
        self.db.setdefault(name, []).append(emb)

    def recognize(self, emb):
        if not self.db:
            return "Banco vazio", 0.0
        q    = emb.reshape(1, -1)
        best_name, best_score = "Desconhecido", 0.0
        for name, embs in self.db.items():
            avg   = np.mean(embs, axis=0).reshape(1, -1)
            score = cosine_similarity(q, avg)[0][0]
            if score > best_score:
                best_score, best_name = score, name
        if best_score < self.threshold:
            return "Desconhecido", best_score
        return best_name, best_score

    def save(self, path):
        np.savez(path, **{k: np.array(v) for k, v in self.db.items()})
        print(f"💾  Banco salvo em '{path}' ({len(self.db)} identidade(s)).")

    def load(self, path):
        if not os.path.exists(path):
            return
        data   = np.load(path)
        self.db = {k: list(data[k]) for k in data.files}
        print(f"📂  Banco carregado: {list(self.db.keys())}")

    def __len__(self):
        return sum(len(v) for v in self.db.values())


# ══════════════════════════════════════════════
# 4. DETECÇÃO DE FACES
# ══════════════════════════════════════════════

_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces(gray, scale=1.1, neighbors=5, min_size=(50, 50)):
    faces = _cascade.detectMultiScale(gray, scale, neighbors, minSize=min_size)
    return faces if len(faces) > 0 else []


# ══════════════════════════════════════════════
# 5. MODO CADASTRO
# ══════════════════════════════════════════════

def register_mode(model, device, db):
    """
    Abre a câmera e permite cadastrar identidades interativamente.
    Pressione ENTER para capturar, ESC para sair.
    """
    print("\n─── MODO CADASTRO ───────────────────────────────────")
    print("  ENTER  → capturar e cadastrar face detectada")
    print("  ESC    → salvar banco e sair\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌  Câmera não encontrada.")
        return

    name = input("Nome da pessoa a cadastrar: ").strip()
    if not name:
        print("Nome vazio, abortando.")
        cap.release()
        return

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        disp  = frame.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 220, 0), 2)

        cv2.putText(disp, f"Cadastrando: {name}  |  amostras: {count}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        cv2.putText(disp, "ENTER=capturar  ESC=sair",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("VGGFace — Cadastro", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break
        elif key == 13 and len(faces) > 0:  # ENTER
            x, y, w, h = faces[0]
            # margem de 20% para melhorar o crop
            pad  = int(0.2 * max(w, h))
            x1   = max(0, x - pad);  y1 = max(0, y - pad)
            x2   = min(frame.shape[1], x + w + pad)
            y2   = min(frame.shape[0], y + h + pad)
            crop = frame[y1:y2, x1:x2]
            emb  = get_embedding(crop, model, device)
            db.register(name, emb)
            count += 1
            print(f"  ✅  Amostra {count} de '{name}' cadastrada.")

    cap.release()
    cv2.destroyAllWindows()
    db.save(DB_PATH)


# ══════════════════════════════════════════════
# 6. MODO RECONHECIMENTO EM TEMPO REAL
# ══════════════════════════════════════════════

# Cores (BGR)
COLOR_KNOWN   = (50, 220, 50)
COLOR_UNKNOWN = (50, 50, 220)
COLOR_INFO    = (220, 220, 220)
FONT          = cv2.FONT_HERSHEY_SIMPLEX


def draw_label(frame, text, x, y, bg_color):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y + 2), bg_color, -1)
    cv2.putText(frame, text, (x + 3, y - 3), FONT, 0.6, (255, 255, 255), 2)


def run_realtime(model, device, db):
    print("\n─── RECONHECIMENTO EM TEMPO REAL ────────────────────")
    print(f"  Identidades no banco : {list(db.db.keys()) or 'nenhuma'}")
    print(f"  Limiar de similaridade: {db.threshold}")
    print("  Q      → sair")
    print("  S      → salvar screenshot")
    print("  R      → cadastrar face do frame atual")
    print("  ESPAÇO → pausar / retomar\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌  Câmera não encontrada. Verifique CAMERA_INDEX.")
        return

    # Tenta forçar resolução razoável
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_n      = 0
    paused       = False
    screenshot_n = 0
    last_results = []   # [(x,y,w,h, name, score), ...]
    fps_time     = time.time()
    fps          = 0.0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("⚠  Falha ao ler frame. Encerrando.")
                break

            frame_n += 1

            # ── Detecção + reconhecimento a cada FRAME_SKIP frames ──
            if frame_n % FRAME_SKIP == 0:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                boxes = detect_faces(gray)
                last_results = []

                for (x, y, w, h) in boxes:
                    pad  = int(0.15 * max(w, h))
                    x1   = max(0, x - pad);  y1 = max(0, y - pad)
                    x2   = min(frame.shape[1], x + w + pad)
                    y2   = min(frame.shape[0], y + h + pad)
                    crop = frame[y1:y2, x1:x2]

                    emb          = get_embedding(crop, model, device)
                    name, score  = db.recognize(emb)
                    last_results.append((x, y, w, h, name, score))

                # FPS
                now      = time.time()
                fps      = 1.0 / (now - fps_time + 1e-9)
                fps_time = now

            # ── Desenha resultados ──
            display = frame.copy()

            for (x, y, w, h, name, score) in last_results:
                color = COLOR_KNOWN if name != "Desconhecido" else COLOR_UNKNOWN
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                label = f"{name}  {score:.2f}"
                draw_label(display, label, x, y, color)

            # HUD
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 28),
                        FONT, 0.65, COLOR_INFO, 2)
            cv2.putText(display, f"Faces: {len(last_results)}", (10, 56),
                        FONT, 0.65, COLOR_INFO, 2)
            cv2.putText(display, "Q=sair  S=screenshot  R=cadastrar  ESPACO=pausar",
                        (10, display.shape[0] - 12), FONT, 0.5, COLOR_INFO, 1)

        else:
            # Frame parado com indicador
            cv2.putText(display, "PAUSADO", (display.shape[1]//2 - 60, 45),
                        FONT, 1.2, (0, 180, 255), 3)

        cv2.imshow("VGGFace — Reconhecimento em Tempo Real", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            paused = not paused
            print("⏸  Pausado." if paused else "▶  Retomado.")

        elif key == ord("s"):
            fname = f"screenshot_{screenshot_n:03d}.jpg"
            cv2.imwrite(fname, display)
            print(f"📸  Screenshot salvo: '{fname}'")
            screenshot_n += 1

        elif key == ord("r"):
            # Cadastra a primeira face detectada no frame atual
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = detect_faces(gray)
            if not boxes:
                print("⚠  Nenhuma face detectada para cadastrar.")
            else:
                x, y, w, h = boxes[0]
                pad  = int(0.2 * max(w, h))
                crop = frame[max(0,y-pad):min(frame.shape[0],y+h+pad),
                             max(0,x-pad):min(frame.shape[1],x+w+pad)]
                cv2.imshow("Confirmar cadastro — ENTER=ok  ESC=cancelar", crop)
                k2 = cv2.waitKey(0) & 0xFF
                cv2.destroyWindow("Confirmar cadastro — ENTER=ok  ESC=cancelar")
                if k2 == 13:
                    name_input = input("  Nome para cadastrar: ").strip()
                    if name_input:
                        emb = get_embedding(crop, model, device)
                        db.register(name_input, emb)
                        db.save(DB_PATH)
                        print(f"  ✅  '{name_input}' cadastrado e banco salvo.")

    cap.release()
    cv2.destroyAllWindows()
    print("✅  Encerrado.")


# ══════════════════════════════════════════════
# 7. ENTRY POINT
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VGGFace — Reconhecimento Facial em Tempo Real"
    )
    parser.add_argument(
        "--register", action="store_true",
        help="Abre modo de cadastro de identidades"
    )
    parser.add_argument(
        "--h5", default=H5_PATH,
        help=f"Caminho para o arquivo de pesos .h5 (padrão: {H5_PATH})"
    )
    parser.add_argument(
        "--db", default=DB_PATH,
        help=f"Caminho para o banco de faces .npz (padrão: {DB_PATH})"
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help=f"Limiar de similaridade de cosseno (padrão: {THRESHOLD})"
    )
    parser.add_argument(
        "--camera", type=int, default=CAMERA_INDEX,
        help=f"Índice da câmera (padrão: {CAMERA_INDEX})"
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Verifica arquivo de pesos
    if not os.path.exists(args.h5):
        print(f"❌  Arquivo de pesos não encontrado: '{args.h5}'")
        print("   Edite H5_PATH no script ou use --h5 <caminho>")
        sys.exit(1)

    # Carrega modelo
    model = load_model(args.h5, device)

    # Carrega banco de faces
    db = FaceDatabase(threshold=args.threshold)
    db.load(args.db)

    # Executa modo escolhido
    if args.register:
        register_mode(model, device, db)
    else:
        if len(db) == 0:
            print("⚠  Banco de identidades vazio!")
            print("   Execute primeiro:  python realtime_recognition.py --register\n")
        run_realtime(model, device, db)


if __name__ == "__main__":
    main()