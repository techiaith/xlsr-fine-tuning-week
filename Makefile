default: build

build:
	docker build --rm -t techiaith/huggingface-xlsr-train-cy .

run: 
	docker run --gpus all --name techiaith-huggingface-xlsr-train-cy \
		 -it \
		-v ${PWD}/homedir:/root \
		-v ${PWD}/models:/models \
		-v ${PWD}/python:/usr/src/python \
		techiaith/huggingface-xlsr-train-cy bash

stop:
	-docker stop techiaith-huggingface-xlsr-train-cy
	-docker rm techiaith-huggingface-xlsr-train-cy

clean:
	-docker rmi techiaith/huggingface-xlsr-train-cy
