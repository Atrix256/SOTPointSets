#pragma once

static const float c_pi = 3.14159265359f;

struct float2
{
	float x, y;
};

inline float Dot(const float2& a, const float2& b)
{
	return a.x * b.x + a.y * b.y;
}

inline float Length(const float2& f)
{
	return std::sqrt(Dot(f, f));
}

inline float2 Normalize(const float2& f)
{
	float len = Length(f);

	float2 ret;
	ret.x = f.x / len;
	ret.y = f.y / len;

	return ret;
}

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