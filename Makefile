container-test:
	docker build -t keras-io .
	docker run --rm -d -p 8000:8000 --name keras-io-server keras-io
	sleep 10
	docker exec keras-io-server echo I am alive
	docker kill keras-io-server
