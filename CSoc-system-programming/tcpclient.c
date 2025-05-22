#include<stdio.h>
#include<stdlib.h>
#include<sys/socket.h>
#include<netinet/in.h>
#include<arpa/inet.h>
#include<string.h>
#include<unistd.h>
#define HOST "192.168.1.9"
int main()
{
	int sockfd, status;
	struct sockaddr_in addr;
	char data[128] ;
	char name[12];

	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if(sockfd < 0)
	{
		perror("Socket didn't connect\n");
		return sockfd;
	}
	
	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_port = htons(8080);
	inet_aton(HOST, &addr.sin_addr); 

	status = connect(sockfd, (struct sockaddr *) &addr, sizeof(addr));
	if(status)
	{
		perror("Connection failed.\n");
		close(sockfd);
		return status;
	}
	printf("Enter your name:");
	scanf("%s", name);

	sprintf(data, name);
	send(sockfd, data, strlen(data), 0);

	recv(sockfd, data, strlen(data), 0);
	close(sockfd);
}
