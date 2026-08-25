## map free inference with map-anything
```shell
uv run python scripts/map_free_inference.py --image-list-csv /home/kobayashi/research-posenet/mapfree_train_all_scenes_interval10_image_paths.csv --long-side-resolution 714 --num-images 24
```


## map free inference with pi3x
```shell
uv sync --extra pi3
```

```shell
uv run python scripts/map_free_inference.py --image-list-csv /home/kobayashi/research-posenet/mapfree_train_all_scenes_interval10_image_paths.csv --model pi3x
```