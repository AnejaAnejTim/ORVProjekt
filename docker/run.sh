docker rm face-server
docker build -t face-id-server .
docker run --name face-server -p 5001:5001 face-id-server
