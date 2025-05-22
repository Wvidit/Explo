#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include<sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

int main()
{
	int sockfd, status;
	struct sockaddr_in addr;
	char body[12];

	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if( sockfd <0)
	{
		perror("Failed to connect to socket.");
		return sockfd;
	}

	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_port = htons(8080);
	addr.sin_addr.s_addr = INADDR_ANY;

	status = bind(sockfd, (struct sockaddr*) &addr, sizeof(addr));
	if(status)
	{
		perror("Not able to bind");
		close(sockfd);
		return status;
	}

	
	status = listen(sockfd, 1);
	if(status)
	{
		perror("Didn't able to listen.");
		close(sockfd);
		return status;
	}
	
	//Conneting client to my socket
	int clientfd;
	struct sockaddr_in client_addr;
	socklen_t addr_len = sizeof(client_addr);
	char text[225];

	clientfd = accept(sockfd, (struct sockaddr*) &client_addr, &addr_len);
	printf("Client Port: %d Client IP: %s", client_addr.sin_port, client_addr.sin_addr.s_addr); 
		
	if(clientfd)
		printf("Problem with client connection.");

	if(recv(clientfd, text, strlen(text), 0))
	{
	printf("Received succesfully.");
	sprintf(text, strcat("hello", text));
	send(clientfd, text, strlen(text), 0);
	}
	close(clientfd);
	close(sockfd);

}
	
