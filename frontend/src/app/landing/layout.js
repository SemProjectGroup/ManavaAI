"use client";
import React from "react";
import { usePathname, useSearchParams } from "next/navigation";

const layout = (props) => {
  const pathname = usePathname();
  console.log(pathname);
  console.log(pathname);

  return <div>{props.children}</div>;
};

export default layout;
