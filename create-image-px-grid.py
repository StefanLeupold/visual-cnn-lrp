import json

px = 32
start_x = 155 + 80
start_y = -415

initial = {
		"id":9178,
		"x":100,
		"y":550,
		"z":0,
		"size":2,
		"type":"heatmap",
		"layerNum": 8
}

all = []

for i in range(px):
	for j in range(px):
		initial = initial.copy()
		initial['id'] = initial['id'] + 1
		initial['x'] = start_x + j*10 
		initial['y'] = start_y - i*10

		all.append(initial)

print(json.dumps(all, indent=4))

