"""
Fabrica de Videos - servidor MCP
--------------------------------
Faz o que a pagina do Claude nao consegue fazer sozinha:
  * narracao com Edge TTS (MP3) + legenda SRT sincronizada palavra a palavra
  * imagens por cena com a API do Gemini (ou imagens enviadas por voce, ex.: Google Flow)
  * montagem do video final com ffmpeg (imagens + narracao + legenda queimada)

Rode localmente:   python server.py
Variaveis de ambiente:
  MCP_TOKEN       segredo que vira parte da URL do conector (obrigatorio em producao)
  PUBLIC_URL      URL publica do servidor, ex.: https://fabrica.onrender.com
  GEMINI_API_KEY  chave do Google AI Studio (so para gerar imagens automaticamente)
  GEMINI_IMAGE_MODEL  padrao: gemini-2.5-flash-image
  FONTE_LEGENDA   padrao: DejaVu Sans (no Docker) / Arial (Windows)
  PORT            padrao: 8000
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import edge_tts
import httpx
from PIL import Image
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse

TOKEN = os.environ.get("MCP_TOKEN") or "dev"
PORT = int(os.environ.get("PORT", "8000"))
PUBLIC_URL = (
    os.environ.get("PUBLIC_URL")
    or (f"https://{os.environ['SPACE_HOST']}" if os.environ.get("SPACE_HOST") else "")  # Hugging Face Spaces
    or os.environ.get("RENDER_EXTERNAL_URL")  # Render
    or f"http://localhost:{PORT}"
).rstrip("/")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
FONTE = os.environ.get("FONTE_LEGENDA") or ("Arial" if os.name == "nt" else "DejaVu Sans")
DADOS = Path(os.environ.get("PASTA_DADOS", Path(__file__).parent / "trabalhos"))
DADOS.mkdir(parents=True, exist_ok=True)

# ESCALA_VIDEO=0.667 gera 720p (bem mais rapido em servidores fracos, como o Render gratis)
ESCALA = float(os.environ.get("ESCALA_VIDEO", "1"))
_par = lambda n: int(n * ESCALA) // 2 * 2  # noqa: E731
FORMATOS = {k: (_par(w), _par(h)) for k, (w, h) in {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080)}.items()}
FPS = int(os.environ.get("FPS_VIDEO", "30"))
PRESET = os.environ.get("PRESET_X264", "veryfast")

mcp = FastMCP(
    "Fabrica de Videos",
    instructions="Gera narracao (Edge TTS) com SRT, imagens por cena e o video final legendado.",
    host="0.0.0.0",
    port=PORT,
    streamable_http_path=f"/{TOKEN}/mcp",
    stateless_http=True,
    json_response=True,
)


# --------------------------------------------------------------------------- util

def _pasta(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{24}", job_id or ""):
        raise ValueError("job_id invalido")
    p = DADOS / job_id
    if not p.exists():
        raise ValueError("job_id nao encontrado (o servidor pode ter reiniciado; gere a narracao de novo)")
    return p


def _url(job_id: str, nome: str) -> str:
    return f"{PUBLIC_URL}/arquivos/{job_id}/{nome}?v={int(time.time())}"


def _duracao_ms(arquivo: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(arquivo)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(float(out) * 1000)


def _ts_srt(ms: int) -> str:
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _ts_ass(ms: int) -> str:
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h}:{m:02}:{s:02}.{ms // 10:02}"


def _agrupar(palavras: list[dict], por_bloco: int) -> list[dict]:
    """Junta palavras em blocos curtos de legenda.
    Quebra tambem na troca de cena e nas pausas da fala (o Edge TTS nao manda a pontuacao)."""
    blocos, atual = [], []
    for i, p in enumerate(palavras):
        atual.append(p)
        prox = palavras[i + 1] if i + 1 < len(palavras) else None
        pausa = prox is not None and (prox["cena"] != p["cena"] or prox["inicio"] - p["fim"] > 250)
        if len(atual) >= por_bloco or pausa:
            blocos.append(atual)
            atual = []
    if atual:
        blocos.append(atual)
    saida = []
    for i, b in enumerate(blocos):
        inicio = b[0]["inicio"]
        fim = b[-1]["fim"]
        # emenda com o proximo bloco para a legenda nao piscar
        if i + 1 < len(blocos):
            prox = blocos[i + 1][0]["inicio"]
            if prox - fim < 400:
                fim = prox
        saida.append({"inicio": inicio, "fim": fim, "texto": " ".join(w["texto"] for w in b), "palavras": b})
    return saida


async def _sintetizar(texto: str, voz: str, velocidade: str, tom: str, destino: Path) -> list[dict]:
    com = edge_tts.Communicate(texto, voz, rate=velocidade, pitch=tom, boundary="WordBoundary")
    palavras = []
    with open(destino, "wb") as f:
        async for chunk in com.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                ini = chunk["offset"] // 10_000  # 100ns -> ms
                palavras.append({"texto": chunk["text"], "inicio": ini, "fim": ini + chunk["duration"] // 10_000})
    return palavras


def _ler_meta(pasta: Path) -> dict:
    return json.loads((pasta / "meta.json").read_text("utf-8"))


def _salvar_meta(pasta: Path, meta: dict) -> None:
    (pasta / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), "utf-8")


def _previa(img: Image.Image, largura: int = 360) -> str:
    im = img.convert("RGB")
    im.thumbnail((largura, largura * 2))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- tools

@mcp.tool()
async def listar_vozes(idioma: str = "pt-BR") -> list[dict]:
    """Lista as vozes do Edge TTS de um idioma (ex.: pt-BR, pt-PT, en-US, es-ES)."""
    vozes = await edge_tts.list_voices()
    return [
        {"id": v["ShortName"], "genero": v["Gender"], "idioma": v["Locale"]}
        for v in vozes
        if v["Locale"].lower().startswith(idioma.lower())
    ]


@mcp.tool()
async def gerar_narracao(
    cenas: list[str],
    voz: str = "pt-BR-AntonioNeural",
    velocidade: str = "+0%",
    tom: str = "+0Hz",
    palavras_por_legenda: int = 4,
) -> dict:
    """Gera a narracao com Edge TTS a partir do texto de cada cena.

    Devolve job_id, o tempo de inicio/fim de cada cena, o SRT completo,
    o MP3 em base64 (para ouvir na pagina) e o link publico do MP3/SRT.
    velocidade: ex. "+10%" ou "-5%". tom: ex. "+0Hz" ou "-2Hz".
    """
    cenas = [c.strip() for c in cenas if c and c.strip()]
    if not cenas:
        raise ValueError("Envie pelo menos uma cena com texto.")
    job_id = secrets.token_hex(12)
    pasta = DADOS / job_id
    pasta.mkdir(parents=True)

    resultados = await asyncio.gather(
        *[_sintetizar(t, voz, velocidade, tom, pasta / f"cena_{i:02}.mp3") for i, t in enumerate(cenas)]
    )

    # junta as cenas num unico MP3 e calcula o deslocamento de cada uma
    tempos, palavras, cursor = [], [], 0
    for i, pals in enumerate(resultados):
        dur = _duracao_ms(pasta / f"cena_{i:02}.mp3")
        tempos.append({"cena": i, "inicio_ms": cursor, "fim_ms": cursor + dur})
        palavras += [{**p, "inicio": p["inicio"] + cursor, "fim": p["fim"] + cursor, "cena": i} for p in pals]
        cursor += dur

    entradas = []
    for i in range(len(cenas)):
        entradas += ["-i", f"cena_{i:02}.mp3"]
    filtro = "".join(f"[{i}:a]" for i in range(len(cenas))) + f"concat=n={len(cenas)}:v=0:a=1[a]"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *entradas, "-filter_complex", filtro, "-map", "[a]",
         "-c:a", "libmp3lame", "-b:a", "128k", "narracao.mp3"],
        cwd=pasta, check=True,
    )

    blocos = _agrupar(palavras, max(1, min(12, palavras_por_legenda)))
    srt = "\n".join(
        f"{n}\n{_ts_srt(b['inicio'])} --> {_ts_srt(b['fim'])}\n{b['texto']}\n" for n, b in enumerate(blocos, 1)
    )
    (pasta / "legendas.srt").write_text(srt, "utf-8")
    _salvar_meta(pasta, {"cenas": cenas, "tempos": tempos, "blocos": blocos, "voz": voz, "imagens": {}})

    audio = (pasta / "narracao.mp3").read_bytes()
    return {
        "job_id": job_id,
        "duracao_ms": cursor,
        "tempos": tempos,
        "srt": srt,
        "audio_base64": base64.b64encode(audio).decode(),
        "audio_url": _url(job_id, "narracao.mp3"),
        "srt_url": _url(job_id, "legendas.srt"),
    }


@mcp.tool()
async def gerar_imagem(job_id: str, cena: int, prompt: str, formato: str = "9:16") -> dict:
    """Gera a imagem de uma cena com a API do Gemini e guarda no trabalho.

    formato: "9:16", "16:9" ou "1:1". Devolve uma previa JPEG em base64 e o link da imagem.
    """
    pasta = _pasta(job_id)
    if not GEMINI_KEY:
        raise ValueError("GEMINI_API_KEY nao configurada no servidor. Envie as imagens manualmente (ex.: do Google Flow).")
    corpo = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": formato}},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    async with httpx.AsyncClient(timeout=120) as cli:
        r = await cli.post(url, json=corpo, headers={"x-goog-api-key": GEMINI_KEY})
    if r.status_code != 200:
        raise ValueError(f"Gemini respondeu {r.status_code}: {r.text[:400]}")
    dados = None
    for cand in r.json().get("candidates", []):
        for parte in cand.get("content", {}).get("parts", []):
            if "inlineData" in parte:
                dados = base64.b64decode(parte["inlineData"]["data"])
                break
    if not dados:
        raise ValueError("O Gemini nao devolveu imagem (pode ter bloqueado o prompt). Ajuste o prompt e tente de novo.")
    return _guardar_imagem(pasta, job_id, cena, dados)


@mcp.tool()
async def enviar_imagem(job_id: str, cena: int, imagem_base64: str) -> dict:
    """Guarda uma imagem enviada por voce (PNG/JPEG em base64) como imagem da cena.
    Use para imagens criadas no Google Flow, Canva ou qualquer outro lugar."""
    pasta = _pasta(job_id)
    dados = base64.b64decode(imagem_base64.split(",")[-1])
    return _guardar_imagem(pasta, job_id, cena, dados)


def _guardar_imagem(pasta: Path, job_id: str, cena: int, dados: bytes) -> dict:
    img = Image.open(io.BytesIO(dados))
    nome = f"img_{cena:02}_{uuid.uuid4().hex[:6]}.jpg"
    img.convert("RGB").save(pasta / nome, "JPEG", quality=92)
    meta = _ler_meta(pasta)
    meta["imagens"][str(cena)] = nome
    _salvar_meta(pasta, meta)
    return {"cena": cena, "previa_base64": _previa(img), "url": _url(job_id, nome)}


# --------------------------------------------------------------------------- video

_status: dict[str, dict] = {}


def _ass(blocos: list[dict], w: int, h: int, estilo: str) -> str:
    tam = int(w * (0.075 if w < h else 0.05))
    margem = int(h * (0.22 if w < h else 0.08))
    # no karaoke (\k) a palavra comeca na cor secundaria e passa para a primaria quando e falada
    primaria = "&H0000E5FF" if estilo == "karaoke" else "&H00FFFFFF"  # amarelo em BGR
    linhas = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 0", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Padrao,{FONTE},{tam},{primaria},&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,"
        f"{max(3, tam // 12)},2,2,{int(w * 0.08)},{int(w * 0.08)},{margem},1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for b in blocos:
        if estilo == "karaoke":
            # cada palavra acende em amarelo quando e falada
            partes, t = [], b["inicio"]
            for p in b["palavras"]:
                if p["inicio"] > t:
                    partes.append(f"{{\\k{(p['inicio'] - t) // 10}}}")
                partes.append(f"{{\\k{max(1, (p['fim'] - p['inicio']) // 10)}}}{p['texto'].upper()} ")
                t = p["fim"]
            texto = "".join(partes).strip()
            linhas.append(f"Dialogue: 0,{_ts_ass(b['inicio'])},{_ts_ass(b['fim'])},Padrao,,0,0,0,,{texto}")
        else:
            texto = b["texto"].upper() if estilo != "simples" else b["texto"]
            linhas.append(f"Dialogue: 0,{_ts_ass(b['inicio'])},{_ts_ass(b['fim'])},Padrao,,0,0,0,,{texto}")
    return "\n".join(linhas) + "\n"


def _renderizar(job_id: str, formato: str, estilo: str, movimento: bool) -> None:
    st = _status[job_id]
    try:
        pasta = DADOS / job_id
        meta = _ler_meta(pasta)
        w, h = FORMATOS.get(formato, FORMATOS["9:16"])
        tempos = meta["tempos"]
        faltando = [t["cena"] for t in tempos if str(t["cena"]) not in meta["imagens"]]
        if faltando:
            raise ValueError(f"Faltam imagens das cenas: {', '.join(str(c + 1) for c in faltando)}")

        # 1) um clipe por cena (imagem cobrindo o quadro + zoom lento)
        clipes = []
        for n, t in enumerate(tempos):
            st.update(etapa=f"Cena {n + 1} de {len(tempos)}", progresso=int(80 * n / len(tempos)))
            dur = (t["fim_ms"] - t["inicio_ms"]) / 1000
            quadros = max(1, round(dur * FPS))
            img = meta["imagens"][str(t["cena"])]
            cobrir = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1"
            if movimento:
                z = "min(zoom+0.0007,1.15)" if n % 2 == 0 else "if(eq(on,0),1.15,max(zoom-0.0007,1))"
                vf = (f"scale={w * 3 // 2 // 2 * 2}:{h * 3 // 2 // 2 * 2}:force_original_aspect_ratio=increase,crop={w * 3 // 2 // 2 * 2}:{h * 3 // 2 // 2 * 2},"
                      f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={quadros}:s={w}x{h}:fps={FPS},setsar=1")
            else:
                vf = cobrir + f",fps={FPS}"
            saida = f"clipe_{n:02}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", img, "-vf", vf, "-frames:v", str(quadros),
                 "-c:v", "libx264", "-preset", PRESET, "-pix_fmt", "yuv420p", "-r", str(FPS), saida],
                cwd=pasta, check=True,
            )
            clipes.append(saida)

        # 2) junta clipes, coloca a narracao e queima a legenda
        st.update(etapa="Juntando e legendando", progresso=85)
        (pasta / "lista.txt").write_text("".join(f"file '{c}'\n" for c in clipes), "utf-8")
        (pasta / "legendas.ass").write_text(_ass(meta["blocos"], w, h, estilo), "utf-8")
        filtro = "subtitles=legendas.ass" if estilo != "sem" else "null"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "lista.txt",
             "-i", "narracao.mp3", "-vf", filtro, "-c:v", "libx264", "-preset", PRESET, "-crf", "21",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", "video.mp4"],
            cwd=pasta, check=True,
        )
        for c in clipes:
            (pasta / c).unlink(missing_ok=True)
        st.update(status="pronto", etapa="Pronto", progresso=100, url=_url(job_id, "video.mp4"))
    except subprocess.CalledProcessError as e:
        st.update(status="erro", etapa="Erro no ffmpeg", erro=str(e))
    except Exception as e:  # noqa: BLE001
        st.update(status="erro", etapa="Erro", erro=str(e))


@mcp.tool()
async def renderizar_video(
    job_id: str, formato: str = "9:16", estilo_legenda: str = "karaoke", movimento: bool = True
) -> dict:
    """Comeca a montar o video final (imagens + narracao + legenda queimada).

    estilo_legenda: "karaoke" (palavra acende), "caixa_alta", "simples" ou "sem".
    Acompanhe com status_video(job_id).
    """
    _pasta(job_id)
    if _status.get(job_id, {}).get("status") == "renderizando":
        return _status[job_id]
    _status[job_id] = {"status": "renderizando", "etapa": "Preparando", "progresso": 0}
    threading.Thread(target=_renderizar, args=(job_id, formato, estilo_legenda, movimento), daemon=True).start()
    return _status[job_id]


@mcp.tool()
async def status_video(job_id: str) -> dict:
    """Mostra o andamento da montagem do video. Quando status = "pronto", traz o link do MP4."""
    _pasta(job_id)
    return _status.get(job_id, {"status": "parado", "etapa": "Nao iniciado", "progresso": 0})


# --------------------------------------------------------------------------- rotas HTTP

@mcp.custom_route("/arquivos/{job_id}/{nome}", methods=["GET"])
async def arquivo(req: Request):
    job_id, nome = req.path_params["job_id"], req.path_params["nome"]
    if not re.fullmatch(r"[a-f0-9]{24}", job_id) or not re.fullmatch(r"[\w.-]+", nome):
        return PlainTextResponse("nao encontrado", 404)
    caminho = DADOS / job_id / nome
    if not caminho.is_file():
        return PlainTextResponse("nao encontrado", 404)
    return FileResponse(caminho, filename=nome if nome.endswith((".mp4", ".srt")) else None)


@mcp.custom_route("/", methods=["GET"])
async def saude(_: Request):
    return JSONResponse({"ok": True, "servico": "Fabrica de Videos", "ffmpeg": bool(shutil.which("ffmpeg"))})


# --------------------------------------------------------------------------- login OAuth automatico
# O botao "Vincular" do claude.ai sempre faz um login OAuth. Este servidor aprova sozinho,
# sem tela: quem protege o acesso e o MCP_TOKEN que ja esta na URL do conector.

def _base(req: Request) -> str:
    return PUBLIC_URL if not PUBLIC_URL.startswith("http://localhost") else str(req.base_url).rstrip("/")


@mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
@mcp.custom_route("/.well-known/oauth-protected-resource/{resto:path}", methods=["GET"])
async def oauth_recurso(req: Request):
    b = _base(req)
    return JSONResponse({"resource": f"{b}/{TOKEN}/mcp", "authorization_servers": [b],
                         "bearer_methods_supported": ["header"]})


@mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
@mcp.custom_route("/.well-known/oauth-authorization-server/{resto:path}", methods=["GET"])
@mcp.custom_route("/.well-known/openid-configuration", methods=["GET"])
async def oauth_servidor(req: Request):
    b = _base(req)
    return JSONResponse({
        "issuer": b,
        "authorization_endpoint": f"{b}/authorize",
        "token_endpoint": f"{b}/token",
        "registration_endpoint": f"{b}/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_post", "client_secret_basic"],
        "scopes_supported": ["mcp"],
    })


@mcp.custom_route("/register", methods=["POST"])
async def oauth_registro(req: Request):
    try:
        dados = await req.json()
    except Exception:  # noqa: BLE001
        dados = {}
    return JSONResponse({
        "client_id": secrets.token_hex(16),
        "client_id_issued_at": int(time.time()),
        "redirect_uris": dados.get("redirect_uris", []),
        "client_name": dados.get("client_name", "cliente"),
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }, status_code=201)


@mcp.custom_route("/authorize", methods=["GET"])
async def oauth_autorizar(req: Request):
    q = req.query_params
    destino = q.get("redirect_uri", "")
    if not destino.startswith("https://") and not destino.startswith("http://localhost"):
        return PlainTextResponse("redirect_uri invalido", 400)
    from urllib.parse import urlencode
    params = {"code": secrets.token_urlsafe(24)}
    if q.get("state"):
        params["state"] = q["state"]
    sep = "&" if "?" in destino else "?"
    return RedirectResponse(f"{destino}{sep}{urlencode(params)}", status_code=302)


@mcp.custom_route("/token", methods=["POST"])
async def oauth_token(_: Request):
    return JSONResponse({
        "access_token": secrets.token_urlsafe(32),
        "token_type": "Bearer",
        "expires_in": 60 * 60 * 24 * 365,
        "refresh_token": secrets.token_urlsafe(32),
        "scope": "mcp",
    }, headers={"Cache-Control": "no-store"})


def _limpar_antigos() -> None:
    """Apaga trabalhos com mais de 2 dias para nao lotar o disco."""
    limite = time.time() - 2 * 86400
    for p in DADOS.iterdir():
        if p.is_dir() and p.stat().st_mtime < limite:
            shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    _limpar_antigos()
    print(f"Servidor no ar em {PUBLIC_URL} (conector: {PUBLIC_URL}/<MCP_TOKEN>/mcp)")
    mcp.run(transport="streamable-http")
