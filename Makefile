# Makefile
build:
	docker-compose build

start:
	docker-compose up -d

stop:
	docker-compose down -v

restart:
	make stop
	make start
	make logs

logs:
	docker-compose logs -f faceid_58

rebuilt:
	make stop
	make build
	make start
