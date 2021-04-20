default: build

build:
	docker build --rm -t techiaith/huggingface-xlsr-train-cy-${USER} .

run: 
	docker run --gpus all --name techiaith-huggingface-xlsr-train-cy-${USER} \
		 -it \
		-v ${PWD}/homedir:/root \
		-v ${PWD}/models:/models \
		-v ${PWD}/python:/usr/src/xlsr-finetune \
		techiaith/huggingface-xlsr-train-cy-${USER} bash

stop:
	-docker stop techiaith-huggingface-xlsr-train-cy-${USER}
	-docker rm techiaith-huggingface-xlsr-train-cy-${USER}

clean:
	-docker rmi techiaith/huggingface-xlsr-train-cy-${USER}
	sudo rm -rf homedir
	mkdir -p homedir
