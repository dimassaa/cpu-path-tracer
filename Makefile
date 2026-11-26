CC      ?= gcc
CFLAGS  ?= -O3 -march=native -ffast-math -Wall -Wextra
LDLIBS  ?= -lpthread -lm

pt: pt.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

cornell.png: pt
	./pt --width 900 --height 900 --spp 1600 -o $@

studio.png: pt
	./pt --scene open --width 900 --height 900 --spp 1200 -o $@

materials.png: pt
	./pt --scene materials --width 900 --height 720 --spp 1600 -o $@

turntable.gif: pt
	mkdir -p frames
	@for i in $$(seq 0 59); do \
		./pt --scene open --angle $$((i * 6)) --width 360 --height 360 --spp 128 \
			-o frames/f$$(printf %03d $$i).png 2>/dev/null; \
	done
	python3 -c "from PIL import Image; import glob; \
f = sorted(glob.glob('frames/f*.png')); \
im = [Image.open(p).convert('RGB') for p in f]; \
im[0].save('turntable.gif', save_all=True, append_images=im[1:], duration=70, loop=0, optimize=True)"

check: pt
	./pt --scene furnace --width 200 --height 200 --spp 4000 --depth 20 --stats -o /tmp/furnace.png

clean:
	rm -f pt
	rm -rf frames

.PHONY: clean check
