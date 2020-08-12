# Makefile
build:
	docker-compose -f docker-compose.dev.yml build

start:
	docker-compose -f docker-compose.dev.yml up -d

stop:
	docker-compose -f docker-compose.dev.yml down -v

restart:
	make stop
	make start
	make logs

logs:
	docker-compose -f docker-compose.dev.yml logs -f

rebuilt:
	make stop
	make build
	make start
