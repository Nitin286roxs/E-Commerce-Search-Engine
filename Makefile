image_name=ecom-se

buildd:
	docker build --target prod -f docker/Dockerfile -t ${image_name}:prod .

