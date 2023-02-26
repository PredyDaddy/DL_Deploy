#include "test.hpp"

#define Title    "This is "
#define Add(a, b)   a + b

int main()
{
  const char *text = Title "Computer Vision";
  int sum = Add(5, 9);
  return 0;
}