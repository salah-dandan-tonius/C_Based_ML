#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE_LEN 8192

void extract_str(const char *line, const char *key, char *out, size_t maxlen) {
    char *pos = strstr(line, key);
    if (!pos) {
        out[0] = '\0';
        return;
    }

    pos = strchr(pos, ':');
    if (!pos) {
        out[0] = '\0';
        return;
    }

    pos++;

    while (*pos == ' ' || *pos == '\"') pos++;

    char *end = pos;
    while (*end && *end != '\"' && *end != ',' && *end != '}') end++;

    size_t len = end - pos;
    if (len >= maxlen) len = maxlen - 1;

    strncpy(out, pos, len);
    out[len] = '\0';
}

int extract_int(const char *line, const char *key) {
    char buf[64];
    extract_str(line, key, buf, sizeof(buf));
    if (buf[0] == '\0') return 0;
    return atoi(buf);
}

int main() {
    FILE *in = fopen("Data/orion-pipeline-2025-01-20.00.json", "r");
    FILE *out = fopen("output.json", "w");
    if (!in || !out) {
        perror("File open error");
        return 1;
    }

    char line[LINE_LEN];
    while (fgets(line, sizeof(line), in)) {
        char src_ip[64], prefix[64], tcp[16], icmp[16];

        extract_str(line, "\"SourceIP\"", src_ip, sizeof(src_ip));
        extract_str(line, "\"Prefix\"", prefix, sizeof(prefix));
        extract_str(line, "\"TCP\"", tcp, sizeof(tcp));
        extract_str(line, "\"ICMP\"", icmp, sizeof(icmp));

        int port = extract_int(line, "\"Port\"");
        int traffic = extract_int(line, "\"Traffic\"");
        int packets = extract_int(line, "\"Packets\"");
        int bytes = extract_int(line, "\"Bytes\"");
        int unique_dests = extract_int(line, "\"UniqueDests\"");
        int unique_dest24s = extract_int(line, "\"UniqueDest24s\"");
        int asn = extract_int(line, "\"ASN\"");

        fprintf(out, "{\"SourceIP\":\"%s\",\"Port\":%d,\"Traffic\":%d,\"Packets\":%d,"
                     "\"Bytes\":%d,\"UniqueDests\":%d,\"UniqueDest24s\":%d,\"ASN\":%d,"
                     "\"Prefix\":\"%s\",\"TCP\":\"%s\",\"ICMP\":\"%s\"}\n",
                src_ip, port, traffic, packets, bytes,
                unique_dests, unique_dest24s, asn, prefix, tcp, icmp);
    }

    fclose(in);
    fclose(out);

    return 0;
}