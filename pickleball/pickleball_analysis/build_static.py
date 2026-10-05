"""Export the current viewer and analyzed clips for GitHub Pages (no backend)."""
import argparse
import json
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
DEFAULT_DATA = Path('/Users/chriscremer/Downloads/pickleball_video_analysis')


def build(data: Path, output: Path):
    data, output = data.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    web = PROJECT / 'web'
    html = (web / 'index.html').read_text()
    html = html.replace('href="/style.css"', 'href="./style.css"')
    html = html.replace('"three":"/vendor/three.module.js"', '"three":"./vendor/three.module.js"')
    html = html.replace('href="/"', 'href="./index.html"')
    html = html.replace('src="/app.js"', 'src="./app.js"')
    html = html.replace('<body>', '<body data-clips-url="./clips.json" data-media-base="./media/">')
    (output / 'index.html').write_text(html)
    for name in ('app.js', 'style.css'):
        shutil.copy2(web / name, output / name)
    shutil.copytree(web / 'vendor', output / 'vendor', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('.DS_Store'))

    clips = []
    for source in sorted((data / 'analysis').glob('rally-*.json')):
        analysis = json.loads(source.read_text())
        clip = dict(analysis['clip'])
        video_path = clip['video'].removeprefix('/media/')
        video = (data / video_path).resolve()
        if not video.is_relative_to(data) or not video.is_file():
            raise ValueError(f'Clip video is missing or outside the data folder: {video_path}')
        target = output / 'media' / video_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video, target)
        analysis['clip']['video'] = video_path
        target = output / 'media' / 'analysis' / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(analysis, separators=(',', ':')))
        clips.append({'file': source.name, **clip, 'video': video_path})
    if not clips:
        raise ValueError('No analyzed clips to publish')
    (output / 'clips.json').write_text(json.dumps(clips, indent=2) + '\n')
    print(f'Exported {len(clips)} clips and the viewer to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=PROJECT)
    args = parser.parse_args()
    build(args.data, args.output)
