# Builds one command: docker build --build-arg CMD=gateway -f deploy/Dockerfile .
FROM golang:1.25 AS build
ARG CMD
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd cmd
COPY internal internal
RUN mkdir -p /out
RUN CGO_ENABLED=0 go build -trimpath -ldflags=-s -o /out/app ./cmd/${CMD}

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/app /app
ENTRYPOINT ["/app"]
