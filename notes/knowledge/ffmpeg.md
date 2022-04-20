ffmpeg -i input.mkv -c:v libx264 -map 0:v -map 0:a:1 -map 0:s:0 -crf 18 -preset veryslow -tune animation output.mkv

ffmpeg -i input -vf scale=1280:690 -c:v libx264 -crf 18 -preset veryslow -tune animation output.mkv
