## POST

`curl -v --header "Content-Type: application/json" --data '{"name": "James Bond"}' http://localhost:8080/api/v1/person`

`curl -v --header "Content-Type: application/json" --data @local.json http://localhost:8080/api/v1/person`


## GET

`curl -v http://localhost:8080/api/v1/person`


## DELETE

`curl -v -H "Content-Type: application/json" -X DELETE http://localhost:8080/api/v1/person/UUID`


## PUT

`curl -v -H "Content-Type: application/json" -X PUT -d '{"id": ""}' http://localhost:8080/api/v1/person/UUID`
