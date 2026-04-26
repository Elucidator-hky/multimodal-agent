"""生成手册图文对照预览 HTML，方便人工查看"""
import ast
import os
import html
from pathlib import Path

ROOT = Path(__file__).parent
MANUAL_DIR = ROOT / "data" / "KnowledgeBase" / "手册"
IMG_DIR = MANUAL_DIR / "插图"
OUT_HTML = ROOT / "preview_manuals.html"

PICKED = [
    "蓝牙激光鼠标手册.txt",
    "VR头显手册.txt",
    "相机手册.txt",
    "冰箱手册.txt",
    "儿童电动摩托车手册.txt",
    "电钻手册.txt",
    "空调手册.txt",
    "吹风机手册.txt",
]


def find_image(img_id: str) -> str | None:
    """图片实际是 .jpg 后缀，列表里 ID 不带后缀"""
    for ext in (".jpg", ".jpeg", ".png"):
        p = IMG_DIR / f"{img_id}{ext}"
        if p.exists():
            return str(p.relative_to(ROOT)).replace("\\", "/")
    return None


def render_manual(name: str) -> str:
    path = MANUAL_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        data = ast.literal_eval(f.read())
    text, imgs = data[0], data[1]
    parts = text.split("<PIC>")

    pic_count = len(parts) - 1
    out = [f'<section><h2>{html.escape(name)}</h2>']
    out.append(f'<p class="meta">正文 {len(text)} 字 | &lt;PIC&gt; 占位符 {pic_count} 个 | 图片列表 {len(imgs)} 个</p>')

    for i, seg in enumerate(parts):
        seg_html = html.escape(seg).replace("\n", "<br>")
        out.append(f'<div class="text">{seg_html}</div>')
        if i < len(imgs):
            img_id = imgs[i]
            img_path = find_image(img_id)
            if img_path:
                out.append(
                    f'<div class="pic">'
                    f'<div class="cap">第 {i+1} 个 &lt;PIC&gt; → <code>{html.escape(img_id)}</code></div>'
                    f'<img src="{img_path}" alt="{html.escape(img_id)}">'
                    f'</div>'
                )
            else:
                out.append(f'<div class="pic missing">第 {i+1} 个 &lt;PIC&gt; → <code>{html.escape(img_id)}</code> ❌ 图片找不到</div>')
    out.append("</section>")
    return "\n".join(out)


def main():
    sections = [render_manual(n) for n in PICKED]
    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>手册图文对照预览</title>
<style>
  body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; max-width: 900px; margin: 0 auto; padding: 24px; line-height: 1.7; color: #222; background: #fafafa; }}
  h1 {{ border-bottom: 3px solid #4a90e2; padding-bottom: 8px; }}
  section {{ background: white; padding: 20px 28px; margin: 24px 0; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  h2 {{ color: #4a90e2; margin-top: 0; }}
  .meta {{ color: #888; font-size: 13px; margin-top: -6px; }}
  .text {{ background: #f6f8fa; padding: 10px 14px; border-left: 3px solid #ddd; margin: 8px 0; font-size: 14px; white-space: pre-wrap; }}
  .pic {{ margin: 12px 0; padding: 10px; border: 2px dashed #4a90e2; background: #f0f7ff; border-radius: 6px; text-align: center; }}
  .pic.missing {{ border-color: #e74c3c; background: #fff5f5; color: #c0392b; }}
  .cap {{ font-size: 12px; color: #555; margin-bottom: 8px; }}
  .pic img {{ max-width: 100%; max-height: 400px; border: 1px solid #ccc; background: white; }}
  code {{ background: #eef; padding: 1px 6px; border-radius: 3px; font-size: 13px; }}
  nav {{ background: white; padding: 14px 20px; border-radius: 8px; margin-bottom: 20px; }}
  nav a {{ display: inline-block; margin-right: 12px; color: #4a90e2; text-decoration: none; }}
  nav a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>手册图文对照预览</h1>
<p>每段正文后面紧跟该位置的 <code>&lt;PIC&gt;</code> 对应图片。文本被 <code>&lt;PIC&gt;</code> 切片后顺序展示。</p>
<nav>
跳转：{' | '.join(f'<a href="#m{i}">{html.escape(n)}</a>' for i, n in enumerate(PICKED))}
</nav>
{''.join(s.replace('<section>', f'<section id="m{i}">') for i, s in enumerate(sections))}
</body>
</html>
"""
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    print(f"已生成: {OUT_HTML}")
    print(f"共 {len(PICKED)} 份手册")


if __name__ == "__main__":
    main()
