#pragma once

static const float c_pi = 3.14159265359f;

struct float2
{
	float x, y;
};

inline float Lerp(float A, float B, float t)
{
	return A * (1.0f - t) + B * t;
}

template <typename T>
inline float Clamp(T value, T themin, T themax)
{
	if (value <= themin)
		return themin;

	if (value >= themax)
		return themax;

	return value;
}