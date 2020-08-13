# Makefile
build:
	docker-compose -f docker-compose.dev.yml build

up:
	docker-compose -f docker-compose.dev.yml up -d

down:
	docker-compose -f docker-compose.dev.yml down -v

restart:
	make down
	make up
	make logs

logs:
	docker-compose -f docker-compose.dev.yml logs -f

rebuild:
	make down
	make build
	make up
	make logs
